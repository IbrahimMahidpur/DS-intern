"""
Reporter Agent — synthesises all pipeline outputs into a structured
markdown report. This is the final node in the LangGraph StateGraph.

Design:
  - Reads code_outputs, visualizations, errors, eval_report, analysis_plan
  - Calls Ollama to produce a structured narrative report
  - Stores the report in state['final_report'] and writes it to disk
  - Also saves eval_report.json to the session working directory

Report structure:
  1. Executive Summary
  2. Key Findings (numbered, quantitative)
  3. Methodology
  4. Results (with inline chart references)
  5. Evaluation Quality Scores
  6. Limitations
  7. Recommendations
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from multimodal_ds.config import REVIEWER_MODEL, OUTPUT_DIR
from multimodal_ds.core.state import AgentState

logger = logging.getLogger(__name__)

REPORTER_SYSTEM = """You are a senior data scientist writing a final analysis report.

Structure your report EXACTLY as:
# Executive Summary
(2-3 sentences summarising what was done and the main finding)

# Key Findings
1. (quantitative finding with numbers)
2. ...

# Methodology
(what analysis steps were executed)

# Results
(detailed results, reference charts as ![Chart](filename.png))

# Quality Assessment
(summarise evaluation scores if provided)

# Limitations
(data quality issues, model assumptions, caveats)

# Recommendations
(3-5 actionable next steps)

Use markdown. Be precise and quantitative. Reference actual numbers from the outputs."""


def _call_ollama(prompt: str, system: str = REPORTER_SYSTEM) -> str:
    """Call LLM for report generation with OpenCode Zen / Ollama fallback."""
    from multimodal_ds.core.llm_client import chat_with_fallback
    try:
        return chat_with_fallback(
            primary_model=REVIEWER_MODEL,
            fallback_model="ollama/qwen2.5:7b",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=4000,
            temperature=0.2,
        )
    except Exception as e:
        logger.error(f"[Reporter] LLM call failed: {e}")
        return _fallback_report(prompt)


def _fallback_report(context: str) -> str:
    """Generate a minimal structured report when LLM is unavailable."""
    return f"""# Analysis Report

## Executive Summary
Analysis pipeline completed. LLM reporter was unavailable for narrative generation.
Raw outputs are included below for review.

## Raw Pipeline Outputs
```
{context[:3000]}
```

## Recommendations
- Review the raw code outputs above for key findings
- Re-run with the reporter LLM available for a structured narrative
"""


def reporter_agent(state: AgentState) -> AgentState:
    """
    LangGraph node: Generate the final report and save to disk.

    Reads:
      state['code_outputs'], state['visualizations'], state['errors'],
      state['analysis_plan'], state['eval_report'], state['user_query']

    Writes:
      state['final_report']  — markdown string
      {session_dir}/final_report.md  — on disk
      {session_dir}/eval_report.json — on disk
    """
    session_id = state.get("session_id", "default")
    logger.info(f"[Reporter] Generating final report for session {session_id}")

    # Assemble context for the LLM – clearer sections with markdown formatting
    all_outputs = "\n\n".join(state.get("code_outputs", []))
    charts = "\n".join(state.get("visualizations", []))
    artifacts = "\n".join(state.get("saved_artifacts", []))
    errors = "\n".join(state.get("errors", []))
    eval_report = state.get("eval_report", {})
    if not isinstance(eval_report, dict):
        eval_report = {
            "session_verdict": getattr(eval_report, "session_verdict", "UNKNOWN"),
            "overall_session_score": getattr(eval_report, "overall_session_score", "N/A"),
            "evaluations": getattr(eval_report, "evaluations", []),
        }
    query = state.get("user_query", "")

    # Build a clean numbered plan string from analysis_tasks.
    #
    # Why not use state.get("analysis_plan", "")?
    # _planner_node sets analysis_plan = plan_result.get("final_plan", "") which
    # is the human-readable executive summary from create_final_plan() — correct.
    # But decompose_into_tasks() stores the raw task list in state["analysis_plan"]
    # as a Python list, and depending on LangGraph's reducer ordering the reporter
    # may receive the list repr "[{'step': 1, ...}]" instead of the summary string.
    # Reading from analysis_tasks (always a list of dicts) and formatting it here
    # gives the reporter a deterministic, readable numbered list regardless of
    # which value ended up in the analysis_plan string field.
    analysis_tasks = state.get("analysis_tasks", [])
    if analysis_tasks:
        plan_lines = []
        for t in analysis_tasks:
            step  = t.get("step", "?")
            name  = t.get("name", "unnamed")
            ttype = t.get("type", "")
            desc  = t.get("description", "")
            expected = t.get("expected_output", "")
            plan_lines.append(
                f"{step}. [{ttype.upper()}] {name}\n"
                f"   Description: {desc}\n"
                f"   Expected output: {expected}"
            )
        plan = "\n\n".join(plan_lines)
    else:
        # Fall back to the analysis_plan string if tasks list is somehow empty
        raw_plan = state.get("analysis_plan", "No plan recorded.")
        # Guard: if it's a list (Python repr leak), convert to JSON for readability
        if isinstance(raw_plan, list):
            import json as _json
            plan = _json.dumps(raw_plan, indent=2)
        else:
            plan = raw_plan or "No plan recorded."

    # Build evaluation markdown table
    eval_summary = ""
    if eval_report:
        verdict = eval_report.get("session_verdict", "UNKNOWN")
        score = eval_report.get("overall_session_score", "N/A")
        eval_summary = f"**Overall Quality Score:** {score}/10  \
**Verdict:** {verdict}\n\n| Task | Score | Flagged |\n|------|-------|--------|"
        for ev in eval_report.get("evaluations", [])[:5]:
            task_name = ev.get("task_name", "?")
            task_score = ev.get("overall_score", "?")
            flagged = "Yes" if ev.get("flagged") else "No"
            eval_summary += f"\n| {task_name} | {task_score} | {flagged} |"

    
    # ── Build LLM prompt (single assignment — no overwrite) ───────────────
    # Previous code had two prompt assignments; the second silently dropped
    # the eval_summary markdown table that the first had correctly included.
    prompt = f"""Original query: {query}

Analysis plan executed:
{plan}

Step-by-step outputs (first 6000 chars):
{all_outputs[:6000]}

Charts generated (filenames — reference in report as ![title](filename)):
{charts or 'None'}

Production artifacts saved (models, CSVs, JSON):
{artifacts or 'None'}

Evaluation quality summary:
{eval_summary or 'No evaluation data available'}

Errors encountered during execution:
{errors or 'None'}

Write the complete structured analysis report now.
- Reference charts using markdown image syntax: ![Chart Title](filename.html)
- Reference saved models and data files in the Results and Recommendations sections
- Include all quantitative findings from the step-by-step outputs
- Be precise — use actual numbers from the outputs, not placeholders"""

    report = _call_ollama(prompt)

    # ── Persist to disk ───────────────────────────────────────────────────
    session_dir = Path(OUTPUT_DIR) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    report_path = session_dir / "final_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"[Reporter] Report saved → {report_path}")

    # Persist eval_report as JSON
    if eval_report:
        eval_path = session_dir / "eval_report.json"
        eval_path.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
        logger.info(f"[Reporter] Eval report saved → {eval_path}")

    return {"final_report": report}


# ── Standalone entry point (for direct calls from orchestrator) ────────────

def generate_report(
    user_query: str,
    analysis_plan: str,
    code_outputs: list[str],
    visualizations: list[str],
    errors: list[str],
    eval_report: dict,
    session_id: str,
    working_dir: Optional[str] = None,
    # Optional enrichment fields — callers that have them should pass them
    saved_artifacts: Optional[list[str]] = None,
    full_code_outputs: Optional[list[str]] = None,
    analysis_tasks: Optional[list[dict]] = None,
    step_file_map: Optional[dict] = None,
) -> str:
    """
    Callable from AgentOrchestrator without going through the graph.
    Returns the markdown report string and saves it to disk.

    Constructs a complete AgentState-shaped dict including all fields
    added in Phases 8-13. Missing fields caused KeyError inside
    reporter_agent when any new field was accessed — the dict now
    provides safe defaults for every declared AgentState key.
    """
    state: AgentState = {
        # ── Core query fields ─────────────────────────────────────────────
        "user_query":           user_query,
        "uploaded_files":       [],
        "_routing_flags":       {},
        # ── Ingestion outputs ─────────────────────────────────────────────
        "parsed_documents":     [],
        "image_embeddings":     [],
        "audio_transcripts":    [],
        "tabular_summaries":    [],
        "blocked_files":        [],
        # ── Statistical report ────────────────────────────────────────────
        "statistical_report":   {},
        # ── Planning outputs ──────────────────────────────────────────────
        "analysis_plan":        analysis_plan,
        "analysis_tasks":       analysis_tasks or [],
        "hypotheses":           [],
        "web_results":          "",
        "planner_data_context": "",
        # ── Execution state ───────────────────────────────────────────────
        "current_step":         0,
        "steps_total":          0,
        "code_outputs":         code_outputs,
        "full_code_outputs":    full_code_outputs or [],
        "visualizations":       visualizations,
        "saved_artifacts":      saved_artifacts or [],
        "files_created":        [],
        "current_step_files":   [],
        "step_file_map":        step_file_map or {},
        # ── Error tracking ────────────────────────────────────────────────
        "errors":               errors,
        # ── Retry / reflection state ──────────────────────────────────────
        "retry_count":          0,
        "_last_task_name":      "",
        "_last_files_created":  [],
        "_last_success":        True,
        "current_step_success": True,
        # ── Memory / retrieval ────────────────────────────────────────────
        "vector_store_id":      "",
        "retrieved_context":    "",
        # ── Quality gate ──────────────────────────────────────────────────
        "gate_passed":          True,
        "gate_reasons":         [],
        # ── Evaluation ───────────────────────────────────────────────────
        "eval_report":          eval_report,
        # ── Visualisation ─────────────────────────────────────────────────
        "visualization_manifest": "",
        # ── Report ───────────────────────────────────────────────────────
        "final_report":         "",
        # ── Session ──────────────────────────────────────────────────────
        "session_id":           session_id,
        "messages":             [],
    }

    result_state = reporter_agent(state)
    return result_state["final_report"]
