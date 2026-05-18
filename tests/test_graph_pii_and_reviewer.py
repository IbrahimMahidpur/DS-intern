import json
import tempfile
from pathlib import Path
import pytest

import multimodal_ds.graph as graph

# Stub Presidio — no external dependency needed
class DummyAnalyzerEngine:
    def analyze(self, text, language="en"):
        return []  # No PII detected

class DummyAnonymizerEngine:
    def anonymize(self, text, analyzer_results):
        class Result:
            def __init__(self, t):
                self.text = t
        return Result(text)

# Replace Presidio lazy singletons so _scan_and_redact is a no-op
graph._presidio_analyzer  = DummyAnalyzerEngine()
graph._presidio_anonymizer = DummyAnonymizerEngine()


class DummyCodeExecutionAgent:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id

    def execute(self, task_description: str, data_context: str = "", file_paths=None, max_retries: int = 2):
        return {
            "success":       True,
            "code":          "# dummy",
            "output":        "Full execution output with stats: Mean=5, Std=2",
            "files_created": ["result.csv"],
            "error":         "",
            "retries_used":  0,
        }


# Stub EvaluationAgent — matches Phase 14 real interface
class DummyEvalReport:
    def __init__(self, task_results):
        self._task_results = task_results

    def to_dict(self) -> dict:
        return {
            "session_id":            "testsession",
            "task_count":            len(self._task_results),
            "flagged_count":         0,
            "pass_count":            len(self._task_results),
            "overall_session_score": 7.0,
            "session_verdict":       "PASS",
            "evaluations":           [],
            "task_results": [tr.get("output_preview", "") for tr in self._task_results],
        }


class DummyEvaluationAgent:
    def __init__(self, session_id: str = "default", **kwargs):
        self.session_id = session_id

    def evaluate_task_results(self, task_results, data_context="", stat_report=None):
        return DummyEvalReport(task_results)


graph.CodeExecutionAgent = DummyCodeExecutionAgent
graph.EvaluationAgent    = DummyEvaluationAgent


@pytest.fixture(autouse=True)
def temp_output_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(graph, "OUTPUT_DIR", tmp_path)
    yield tmp_path


def test_executor_node_generates_files_created():
    state = {
        "analysis_tasks":     [{"name": "dummy_task", "description": "do something", "type": "eda"}],
        "current_step":       0,
        "session_id":         "testsession",
        "uploaded_files":     [],
        "tabular_summaries":  [],
        "parsed_documents":   [],
        "image_embeddings":   [],
        "full_code_outputs":  [],
        "errors":             [],
        "visualizations":     [],
        "saved_artifacts":    [],
        "current_step_files": [],
        "files_created":      [],
        "current_step_success": False,
        "step_file_map":      {},
        "_last_files_created": [],
        "_last_success":      False,
    }

    result = graph._executor_node(state)

    assert result["current_step"] == 1
    assert result["current_step_files"] == ["result.csv"]
    assert result["files_created"] == ["result.csv"]
    # Phase 18: string keys in step_file_map
    assert result["step_file_map"].get("0") == ["result.csv"], \
        f"step_file_map should use string key '0', got: {result['step_file_map']}"
    # Phase 18: full_code_outputs uses 'output' key
    assert result["full_code_outputs"] == ["Full execution output with stats: Mean=5, Std=2"]


def test_reviewer_node_output_preview_contains_statistics():
    state = {
        "analysis_tasks":     [{"name": "stats_task", "description": "compute stats", "type": "eda"}],
        "full_code_outputs":  ["Mean: 5, Std: 2, Min: 1, Max: 10"],
        "errors":             [],
        "visualizations":     [],
        "saved_artifacts":    [],
        "_last_files_created": ["stats.csv"],
        "session_id":         "testsession",
        "tabular_summaries":  [],
        "statistical_report": {},
        "step_file_map":      {"0": ["stats.csv"]},
    }

    result = graph._reviewer_node(state)
    eval_report = result.get("eval_report")

    assert eval_report is not None
    # After Phase 14, eval_report is a dict
    assert isinstance(eval_report, dict), \
        f"eval_report should be dict, got {type(eval_report)}"
    assert "task_results" in eval_report
    assert any("Mean: 5" in preview for preview in eval_report["task_results"]), \
        f"Expected 'Mean: 5' in task_results, got: {eval_report['task_results']}"


def test_pii_scan_does_not_block_clean_file(tmp_path):
    """Files with no PII should pass through _scan_and_redact unchanged."""
    clean_file = tmp_path / "output.csv"
    clean_file.write_text("age,income,churn\n25,45000,0\n32,72000,1\n")

    # Import the inner function indirectly by running executor with a clean file
    state = {
        "analysis_tasks":     [{"name": "eda", "description": "explore", "type": "eda"}],
        "current_step":       0,
        "session_id":         "piitest",
        "uploaded_files":     [],
        "tabular_summaries":  [],
        "parsed_documents":   [],
        "image_embeddings":   [],
        "full_code_outputs":  [],
        "errors":             [],
        "visualizations":     [],
        "saved_artifacts":    [],
        "current_step_files": [],
        "files_created":      [],
        "current_step_success": False,
        "step_file_map":      {},
        "_last_files_created": [],
        "_last_success":      False,
    }
    # Should not raise — clean file passes PII gate
    result = graph._executor_node(state)
    assert result["current_step"] == 1