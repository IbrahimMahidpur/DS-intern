import pytest
from multimodal_ds.agents.code_execution_agent import CodeExecutionAgent

def test_extract_code_standard_python():
    agent = CodeExecutionAgent()
    text = """
```python
import pandas as pd
print("Hello standard")
```
"""
    code = agent._extract_code(text)
    assert code == 'import pandas as pd\nprint("Hello standard")'

def test_extract_code_windows_line_endings():
    agent = CodeExecutionAgent()
    text = "```python\r\nimport pandas as pd\r\nprint(\"Hello Windows\")\r\n```"
    code = agent._extract_code(text)
    # The output from m.group(1).strip() will keep internal line endings, but standardizes leading/trailing
    assert "import pandas as pd" in code
    assert "Hello Windows" in code
    assert "```" not in code

def test_extract_code_trailing_space():
    agent = CodeExecutionAgent()
    text = "```python  \nimport pandas as pd\nprint(\"Hello Space\")\n```"
    code = agent._extract_code(text)
    assert code == 'import pandas as pd\nprint("Hello Space")'

def test_extract_code_any_fence():
    agent = CodeExecutionAgent()
    text = """
```
import pandas as pd
print("Hello Any")
```
"""
    code = agent._extract_code(text)
    assert code == 'import pandas as pd\nprint("Hello Any")'

def test_extract_code_fallback_raw_text():
    agent = CodeExecutionAgent()
    text = """
import pandas as pd
print("Hello Raw")
"""
    code = agent._extract_code(text)
    assert code == 'import pandas as pd\nprint("Hello Raw")'

def test_extract_code_fallback_raw_text_with_fences_stripped():
    agent = CodeExecutionAgent()
    text = """
```python
import pandas as pd
print("Hello Raw Fenced")
"""
    code = agent._extract_code(text)
    assert code == 'import pandas as pd\nprint("Hello Raw Fenced")'

def test_extract_code_think_tags_stripped():
    agent = CodeExecutionAgent()
    text = """
<think>
This is reasoning.
</think>
```python
import pandas as pd
print("Hello Think")
```
"""
    code = agent._extract_code(text)
    assert code == 'import pandas as pd\nprint("Hello Think")'


def test_is_only_warnings():
    from multimodal_ds.agents.code_execution_agent import _is_only_warnings

    # 1. Clean stderr / Empty
    assert _is_only_warnings("") is True
    assert _is_only_warnings("   \n   ") is True

    # 2. Pure warning
    assert _is_only_warnings("UserWarning: dtypes are wrong") is True
    assert _is_only_warnings(
        "/venv/lib/python3.10/site-packages/pandas/core/frame.py:123: FutureWarning: "
        "The behavior of DataFrame.concat with empty or all-NA entries is deprecated."
    ) is True

    # 3. Real tracebacks/exceptions should return False
    assert _is_only_warnings(
        "Traceback (most recent call last):\n  File \"script.py\", line 5, in <module>\nNameError: name 'x' is not defined"
    ) is False
    assert _is_only_warnings("ZeroDivisionError: division by zero") is False
    assert _is_only_warnings("ValueError: X has 10 features, but Model is expecting 15 features") is False

    # 4. Warnings with no "warning" word but error keyword should return False
    assert _is_only_warnings("failed: connection timed out") is False
