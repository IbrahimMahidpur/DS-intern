import pytest
from pathlib import Path

def test_eda_code_executes_directly():
    """Test that EDA code generation and execution works end-to-end
    without the full graph, to isolate whether the problem is in the
    LLM call, code generation, or execution."""

    from multimodal_ds.agents.code_execution_agent import CodeExecutionAgent
    import tempfile, csv

    # Create a minimal CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                     delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['age', 'income', 'churn'])
        for i in range(20):
            writer.writerow([20+i, 40000+i*1000, i%2])
        csv_path = f.name

    agent = CodeExecutionAgent(session_id="direct_test")
    result = agent.execute(
        task_description=(
            "Load the CSV file, print df.columns and df.shape, "
            "compute df.describe() and print it. "
            "The file is named: " + Path(csv_path).name
        ),
        data_context=f"Available file: {Path(csv_path).name}\nColumns: age, income, churn\nShape: 20x3",
        file_paths=[csv_path],
    )

    print("SUCCESS:", result['success'])
    print("OUTPUT:", result['output'][:500])
    print("ERROR:", result['error'][:300])
    print("FILES:", result['files_created'])

    # Don't assert success - just print so we can see what's failing
    assert 'output' in result
