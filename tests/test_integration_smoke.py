import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from multimodal_ds.graph import make_initial_state, build_graph


class DummyResponse:
    def __init__(self, content: str):
        self.status_code = 200
        self._content = content

    def json(self):
        return {"message": {"content": self._content}}


def dummy_post(url, **kwargs):
    """Return canned Ollama responses for all LLM-calling modules.

    Accepts **kwargs so it works regardless of how httpx.post is called
    (positional json=, timeout=, headers= etc. vary by caller).
    The original dummy_post(url, json=None, timeout=None) signature caused
    TypeError when callers passed unexpected keyword arguments.
    """
    json_body = kwargs.get("json", {})
    messages = json_body.get("messages", []) if isinstance(json_body, dict) else []
    system = messages[0].get("content", "") if messages else ""

    # Hypothesis generation call
    if "hypothes" in system.lower() or "hypothes" in str(json_body).lower():
        content = json.dumps([{
            "id": "h1",
            "statement": "Churn depends on age and income",
            "analysis_method": "logistic regression",
            "expected_outcome": "model predicts churn",
        }])
        return DummyResponse(content)

    # Task decomposition call — return a single minimal EDA task
    if "task" in system.lower() or "plan" in system.lower() or "decompos" in system.lower():
        content = json.dumps([{
            "step": 1,
            "name": "EDA",
            "type": "eda",
            "description": "print(df.columns.tolist()); print(df.shape)",
            "tools": ["pandas"],
            "expected_output": "summary",
            "depends_on": [],
        }])
        return DummyResponse(content)

    # Code generation / fix / eval / reporter / statistical interpreter calls
    # Return a minimal valid Python snippet so code execution has something to run
    if "python" in system.lower() or "code" in system.lower() or "fix" in system.lower():
        content = "```python\nimport pandas as pd\nprint('smoke ok')\n```"
        return DummyResponse(content)

    # Statistical interpretation, evaluation judge, reporter, narrative — return plain text
    return DummyResponse("Analysis complete. No issues found.")


def test_end_to_end_smoke():
    # Create a small CSV with 5 rows and 3 columns
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "data.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["age", "income", "churn"])
            for i in range(5):
                writer.writerow([20 + i, 30000 + i * 1000, i % 2])

        state = make_initial_state(
            user_query="predict churn",
            uploaded_files=[str(csv_path)],
            session_id="smoke_test_session",
        )

        g = build_graph()

        # Patch httpx.post only in the modules that actually call Ollama.
        # A global patch("httpx.post") also intercepts ChromaDB's internal
        # HTTP health checks and LangGraph's SQLite checkpointer, causing
        # those calls to receive a DummyResponse they can't parse and raising
        # AttributeError / JSONDecodeError deep inside those libraries.
        # Narrow the patch to each Ollama-calling module individually.
        ollama_modules = [
            "multimodal_ds.agents.planner_agent.httpx",
            "multimodal_ds.agents.code_execution_agent.httpx",
            "multimodal_ds.agents.statistical_agent.httpx",
            "multimodal_ds.agents.visualization_agent.httpx",
            "multimodal_ds.agents.evaluation_agent.httpx",
            "multimodal_ds.core.llm_client.httpx",
            "multimodal_ds.memory.agent_memory.httpx",
        ]

        patches = [
            patch(f"{mod}.post", side_effect=dummy_post)
            for mod in ollama_modules
        ]

        # Stack all patches and invoke the graph
        with patches[0], patches[1], patches[2], patches[3], \
             patches[4], patches[5], patches[6]:
            final_state = g.invoke(state, config={"configurable": {"thread_id": "smoke_test_thread"}})

        # Core pipeline assertions
        assert final_state.get("tabular_summaries"), \
            "Tabular summaries should be populated after tab_ingest"
        assert final_state.get("statistical_report"), \
            "Statistical report should be populated after stats_val"
        assert final_state.get("analysis_tasks"), \
            "Analysis tasks (planner output) should be non-empty"
        # Clean CSV — no PII columns
        assert not final_state.get("blocked_files"), \
            "There should be no blocked files for clean data"
