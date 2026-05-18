"""
Code Execution Agent — hardened sandbox with resource limits.
FIX: working_dir now includes session_id for session isolation.
"""
from httpx import _status_codes
import logging
import os
import subprocess
import sys
import tempfile
import time
import shutil
import concurrent.futures
from pathlib import Path
from typing import Optional

import httpx

from multimodal_ds.config import CODER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT, OUTPUT_DIR
from multimodal_ds.memory.agent_memory import AgentMemory
from multimodal_ds.core.observability import agent_span, get_session_tracker

logger = logging.getLogger(__name__)

_CPU_SECONDS    = int(os.getenv("SANDBOX_CPU_SECONDS",  "60"))
_MEM_MB         = int(os.getenv("SANDBOX_MEM_MB",       "512"))
_STDOUT_CHARS   = int(os.getenv("SANDBOX_STDOUT_CHARS", "8000"))
_PROC_TIMEOUT_S = int(os.getenv("SANDBOX_TIMEOUT_S",    "300"))

SYSTEM_PROMPT = """You are a senior data scientist. You write precise, self‑contained Python.

MANDATORY RULES — follow every one, no exceptions:
1. First line of code: print(df.columns.tolist()) and print(df.shape)
2. Use ONLY column names confirmed by step 1 — never guess column names
3. Print descriptive stats: df.describe(), value_counts for every categorical column
4. ALWAYS follow this EXACT preprocessing sequence before any model.fit():
   Step A — Identify target and drop useless columns:
       target_col = 'Exited'  # or whatever the target is
       drop_cols = [c for c in df.columns if df[c].nunique() == df.shape[0]]  # IDs
       drop_cols += ['RowNumber', 'CustomerId', 'Surname']  # known non-predictive
       drop_cols = [c for c in drop_cols if c in df.columns]
       df = df.drop(columns=drop_cols, errors='ignore')

   Step B — Encode ALL categoricals with pd.get_dummies (never leave object cols):
       df = pd.get_dummies(df, drop_first=True)

   Step C — Separate features and target AFTER encoding:
       y = df[target_col].astype(int)
       X = df.drop(columns=[target_col])

   Step D — Fill any remaining NaN:
       X = X.fillna(X.median())

   Step E — Verify no object columns (print, do NOT assert):
       obj_cols = X.select_dtypes(include=['object']).columns.tolist()
       if obj_cols:
           print(f'WARNING: dropping remaining object cols: {obj_cols}')
           X = X.drop(columns=obj_cols)
       print(f'Final X shape: {X.shape}, dtypes OK')

   Step F — Split:
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.2, random_state=42, stratify=y)

   NEVER use assert statements — use if/print/drop instead.
   NEVER call model.fit() before completing Steps A-F.
5. ALWAYS set matplotlib backend as the ABSOLUTE FIRST THREE LINES of the entire script,
   before ANY other import including pandas, numpy, sklearn:
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   These 3 lines must appear at line 1, 2, 3 of the file. No exceptions.
   Never import matplotlib.pyplot before calling matplotlib.use('Agg').
6. Save ALL plots: plt.savefig('filename.png', dpi=100, bbox_inches='tight',
   facecolor='white', edgecolor='none')
   Then IMMEDIATELY call plt.close('all') after every savefig call.
   Never reuse a figure across multiple plots.
7. Never call plt.show() under any circumstances.
8. When creating subplot grids with plt.subplots(rows, cols):
   - ALWAYS flatten axes immediately after creation:
     fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
     axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
   - ALWAYS calculate nrows and ncols from actual data AFTER loading:
     n_cols = len(columns_to_plot)
     ncols = min(3, n_cols)
     nrows = (n_cols + ncols - 1) // ncols  # ceiling division
   - ALWAYS iterate with enumerate and check bounds:
     for i, col in enumerate(columns_to_plot):
         if i >= len(axes): break
         axes[i].hist(df[col].dropna(), bins=30)
   - ALWAYS hide unused axes:
     for j in range(i+1, len(axes)): axes[j].set_visible(False)
   - ALWAYS call fig.tight_layout() before savefig
   - NEVER use axes[i] without first calling axes = axes.flatten()
9. Save trained models: joblib.dump(model, 'model.pkl')
10. End with a FINDINGS block: print('=== FINDINGS ===') then 3‑5 quantitative sentences
11. NEVER evaluate a model on training data. Always use held-out test set.
   If cross-validation: use cross_val_score with cv=5 on training data only.
12. Keep ALL string literals on a single line. Never split a string literal across lines using implicit continuation. For long titles use short versions: ax.set_title('H2: Products') not ax.set_title('H2: NumOfProducts > 2 has significantly higher churn rates than customers with 1 or 2 products')
13. NEVER import seaborn before setting matplotlib backend. Import order MUST be:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns  # only after the above
14. NEVER use assert statements in data science code. Assertions crash the script.
    Instead use: if condition: print('warning'); X = X.drop(...)
15. For Churn_Modelling.csv specifically, ALWAYS drop these columns before modeling:
    ['RowNumber', 'CustomerId', 'Surname']
    ALWAYS encode with pd.get_dummies(df, drop_first=True) BEFORE separating X and y.
    The target column is 'Exited'.
16. ALWAYS convert boolean columns to int immediately after pd.get_dummies() or pd.read_csv():
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)
    This prevents numpy.histogram RuntimeWarning and crashes when plotting
17. NEVER wrap code in a def main() function. Write all code at module level only.
    NEVER use 'if __name__ == "__main__":' blocks.
    NEVER reference a variable before it is assigned. Always: imports → load data → process → print results.
18. Your code structure MUST follow this exact order every time:
    1) matplotlib backend (already handled - do NOT add)
    2) All imports
    3) Load data with pd.read_csv()
    4) All processing and modeling
    5) Save files
    6) Print findings
Output only valid Python code inside ```python ... ``` fences. No commentary outside the fences."""

def _strip_main_wrapper(code: str) -> str:
    """
    Handles all patterns where LLM wraps code in main() or if __name__ blocks.
    Unwraps to flat module-level code so the preamble + code works correctly.
    """
    import re
    import textwrap

    # Pattern 1: Remove if __name__ == '__main__': guard entirely
    code = re.sub(
        r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:.*$",
        '',
        code,
        flags=re.MULTILINE,
    )

    # Pattern 2: Unwrap def main(): ... main() into flat code
    main_match = re.search(r'^def main\(\)\s*:\n((?:[ \t]+.*\n?)*)', code, re.MULTILINE)
    if main_match:
        body = main_match.group(1)
        dedented = textwrap.dedent(body)
        # Replace the entire def main() block with dedented body
        code = code[:main_match.start()] + dedented + code[main_match.end():]

    # Pattern 3: Remove any bare main() calls left over
    code = re.sub(r'^\s*main\(\)\s*$', '', code, flags=re.MULTILINE)

    # Pattern 4: Remove consecutive blank lines (cleanup after stripping)
    code = re.sub(r'\n{3,}', '\n\n', code)

    return code.strip()

def _sandbox_preexec() -> None:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS))
        mem_bytes = _MEM_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except Exception:
        pass


def _is_only_warnings(stderr: str) -> bool:
    """Check if stderr contains only warnings (e.g. deprecation/future warnings) and no actual exception traceback or fatal error."""
    if not stderr.strip():
        return True
    stderr_lower = stderr.lower()
    # If a traceback is present, it's a real crash
    if "traceback" in stderr_lower or "stack traceback" in stderr_lower:
        return False
    # Typical error keywords indicating execution failed
    error_keywords = [
        "error:", "exception:", "failed:", "exit status",
        "nameerror", "syntaxerror", "typeerror", "valueerror",
        "keyerror", "indexerror", "attributeerror", "importerror",
        "modulenotfounderror", "zerodivisionerror", "runtimeerror"
    ]
    if any(kw in stderr_lower for kw in error_keywords):
        return False
    # If the stderr has warnings, but no traceback or standard errors, treat as warnings
    return "warning" in stderr_lower


class CodeExecutionAgent:
    AGENT_NAME = "code_execution_agent"

    def __init__(self, working_dir: Optional[str] = None, session_id: str = "default"):
        # FIX: include session_id in working_dir for session isolation
        base = Path(working_dir) if working_dir else Path(OUTPUT_DIR)
        self.working_dir = base / session_id
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.memory = AgentMemory()
        self._tracker = get_session_tracker(session_id)
    
    def _ensure_dependencies(self, code: str) -> None:
        """
        Parse all import statements from generated code and ensure they are
        installed before execution. Uses pip to install missing packages
        automatically. This replaces all hardcoded banned-import lists.
        """
        import ast
        import subprocess
        import sys

        # Built-in modules that should never be pip-installed
        STDLIB_MODULES = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else {
            'os', 'sys', 're', 'json', 'math', 'time', 'datetime', 'pathlib',
            'collections', 'itertools', 'functools', 'operator', 'io', 'abc',
            'copy', 'typing', 'dataclasses', 'enum', 'warnings', 'logging',
            'threading', 'subprocess', 'tempfile', 'shutil', 'glob', 'struct',
            'random', 'hashlib', 'base64', 'urllib', 'http', 'socket', 'ssl',
            'csv', 'configparser', 'argparse', 'ast', 'inspect', 'traceback',
            'contextlib', 'weakref', 'gc', 'platform', 'signal', 'pickle',
            'shelve', 'sqlite3', 'uuid', 'decimal', 'fractions', 'statistics',
            'string', 'textwrap', 'pprint', 'reprlib', 'unicodedata',
        }

        # Package name mapping: import name → pip install name
        # Only add entries where they DIFFER
        IMPORT_TO_PIP = {
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'bs4': 'beautifulsoup4',
            'yaml': 'pyyaml',
            'dotenv': 'python-dotenv',
            'dateutil': 'python-dateutil',
            'pkg_resources': 'setuptools',
            'gi': 'PyGObject',
            'wx': 'wxPython',
            'usaddress': 'usaddress',
            'nltk': 'nltk',
            'spacy': 'spacy',
            'transformers': 'transformers',
            'torch': 'torch',
            'tensorflow': 'tensorflow',
            'keras': 'keras',
            'lightgbm': 'lightgbm',
            'xgboost': 'xgboost',
            'catboost': 'catboost',
            'statsmodels': 'statsmodels',
            'scipy': 'scipy',
            'plotly': 'plotly',
            'bokeh': 'bokeh',
            'altair': 'altair',
            'dash': 'dash',
            'streamlit': 'streamlit',
            'shap': 'shap',
            'lime': 'lime',
            'mlflow': 'mlflow',
            'optuna': 'optuna',
            'hyperopt': 'hyperopt',
            'ax': 'ax-platform',
            'flaml': 'flaml',
            'pycaret': 'pycaret',
            'autosklearn': 'auto-sklearn',
            'tpot': 'tpot',
            'h2o': 'h2o',
            'prophet': 'prophet',
            'pmdarima': 'pmdarima',
            'neuralprophet': 'neuralprophet',
            'lifelines': 'lifelines',
            'pysurvival': 'pysurvival',
            'imbalanced_learn': 'imbalanced-learn',
            'imblearn': 'imbalanced-learn',
            'umap': 'umap-learn',
            'hdbscan': 'hdbscan',
            'pyod': 'pyod',
            'alibi': 'alibi',
            'river': 'river',
            'dask': 'dask',
            'polars': 'polars',
            'vaex': 'vaex',
            'modin': 'modin',
            'numba': 'numba',
            'cupy': 'cupy',
            'pyspark': 'pyspark',
            'sqlalchemy': 'sqlalchemy',
            'pymongo': 'pymongo',
            'redis': 'redis',
            'celery': 'celery',
            'fastapi': 'fastapi',
            'flask': 'flask',
            'django': 'django',
            'requests': 'requests',
            'httpx': 'httpx',
            'aiohttp': 'aiohttp',
            'websockets': 'websockets',
            'paramiko': 'paramiko',
            'cryptography': 'cryptography',
            'jwt': 'PyJWT',
            'bcrypt': 'bcrypt',
            'arrow': 'arrow',
            'pendulum': 'pendulum',
            'pytz': 'pytz',
            'tzdata': 'tzdata',
            'babel': 'Babel',
            'chardet': 'chardet',
            'ftfy': 'ftfy',
            'unidecode': 'Unidecode',
            'regex': 'regex',
            'fuzzywuzzy': 'fuzzywuzzy',
            'Levenshtein': 'python-Levenshtein',
            'gensim': 'gensim',
            'fasttext': 'fasttext',
            'sentence_transformers': 'sentence-transformers',
            'tiktoken': 'tiktoken',
            'openai': 'openai',
            'anthropic': 'anthropic',
            'langchain': 'langchain',
            'langgraph': 'langgraph',
            'chromadb': 'chromadb',
            'faiss': 'faiss-cpu',
            'pinecone': 'pinecone-client',
            'weaviate': 'weaviate-client',
            'qdrant': 'qdrant-client',
            'networkx': 'networkx',
            'igraph': 'python-igraph',
            'graph_tool': 'graph-tool',
            'pyvis': 'pyvis',
            'gephi': 'gephi',
            'folium': 'folium',
            'geopandas': 'geopandas',
            'shapely': 'shapely',
            'pyproj': 'pyproj',
            'rasterio': 'rasterio',
            'opencv': 'opencv-python',
            'skimage': 'scikit-image',
            'imageio': 'imageio',
            'tifffile': 'tifffile',
            'librosa': 'librosa',
            'soundfile': 'soundfile',
            'pydub': 'pydub',
            'pyaudio': 'pyaudio',
            'reportlab': 'reportlab',
            'fpdf': 'fpdf2',
            'docx': 'python-docx',
            'openpyxl': 'openpyxl',
            'xlrd': 'xlrd',
            'xlwt': 'xlwt',
            'xlsxwriter': 'XlsxWriter',
            'tabulate': 'tabulate',
            'prettytable': 'PrettyTable',
            'rich': 'rich',
            'tqdm': 'tqdm',
            'loguru': 'loguru',
            'click': 'click',
            'typer': 'typer',
            'pydantic': 'pydantic',
            'attrs': 'attrs',
            'marshmallow': 'marshmallow',
            'cerberus': 'Cerberus',
            'voluptuous': 'voluptuous',
            'hypothesis': 'hypothesis',
            'pytest': 'pytest',
            'mock': 'mock',
            'faker': 'Faker',
            'factory_boy': 'factory-boy',
            'coverage': 'coverage',
            'mypy': 'mypy',
            'pylint': 'pylint',
            'flake8': 'flake8',
            'black': 'black',
            'isort': 'isort',
            'bandit': 'bandit',
            'safety': 'safety',
            'schedule': 'schedule',
            'apscheduler': 'APScheduler',
            'crontab': 'python-crontab',
            'psutil': 'psutil',
            'memory_profiler': 'memory-profiler',
            'line_profiler': 'line-profiler',
            'py_spy': 'py-spy',
            'objgraph': 'objgraph',
            'joblib': 'joblib',
            'multiprocessing_logging': 'multiprocessing-logging',
            'pathos': 'pathos',
            'dill': 'dill',
            'cloudpickle': 'cloudpickle',
            'zarr': 'zarr',
            'h5py': 'h5py',
            'netCDF4': 'netCDF4',
            'pyarrow': 'pyarrow',
            'fastparquet': 'fastparquet',
            'tables': 'tables',
            'lmdb': 'lmdb',
        }

        # Extract all top-level imports from the generated code
        imported_modules = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_modules.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_modules.add(node.module.split('.')[0])
        except SyntaxError:
            # Code has syntax errors — extract imports with regex as fallback
            import re
            for match in re.finditer(r'^(?:import|from)\s+(\w+)', code, re.MULTILINE):
                imported_modules.add(match.group(1))

        # Filter to only non-stdlib, non-private modules
        to_check = {
            m for m in imported_modules
            if m and not m.startswith('_') and m not in STDLIB_MODULES
        }

        # Check which are missing and install them
        missing = []
        for module in to_check:
            try:
                __import__(module)
            except ImportError:
                pip_name = IMPORT_TO_PIP.get(module, module)
                missing.append(pip_name)

        if missing:
            logger.info(f"[CodeAgent] Auto-installing missing packages: {missing}")
            for pkg in missing:
                try:
                    result = subprocess.run(
                        [sys.executable, '-m', 'pip', 'install', pkg,
                         '--quiet', '--no-warn-script-location'],
                        capture_output=True, text=True, timeout=120,
                    )
                    if result.returncode == 0:
                        logger.info(f"[CodeAgent] Installed: {pkg}")
                    else:
                        logger.warning(
                            f"[CodeAgent] Failed to install {pkg}: {result.stderr[:200]}"
                        )
                except Exception as e:
                    logger.warning(f"[CodeAgent] Install error for {pkg}: {e}")

    def _enforce_matplotlib_backend(self, code: str) -> str:
        """Move matplotlib backend setup to lines 1‑3 if not already there."""
        import re
        backend_block = (
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
        )
        # Check first three lines for correct placement
        first_three = "\n".join(code.splitlines()[:3])
        if "matplotlib.use('Agg')" in first_three:
            return code
        # Remove existing matplotlib imports
        code = re.sub(r'^import matplotlib\n', '', code, flags=re.MULTILINE)
        code = re.sub(r"^matplotlib\.use\(['\"]Agg['\"]\)\n", '', code, flags=re.MULTILINE)
        code = re.sub(r'^import matplotlib\.pyplot as plt\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^from matplotlib import.*\n', '', code, flags=re.MULTILINE)
        return backend_block + code
    
    def _fix_common_code_errors(self, code: str) -> str:
        """
        Universal self-healing sanitizer.
        Applies only mechanical, syntax-level fixes that are always safe.
        """
        import re

        # ── 1. Normalize Unicode to ASCII-safe equivalents ──────────────────
        unicode_map = {
            '\u2011': '-', '\u2012': '-', '\u2013': '-', '\u2014': '-',
            '\u2015': '-', '\u2212': '-', '\u2010': '-', '\ufe58': '-',
            '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u201b': "'",
            '\u2032': "'", '\u2035': "'",
            '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"',
            '\u2033': '"', '\u2036': '"',
            '\u2026': '...', '\u22ef': '...',
            '\u00b7': '*', '\u2022': '*', '\u2023': '*', '\u25e6': '*',
            '\u00a0': ' ', '\u202f': ' ', '\u205f': ' ', '\u3000': ' ',
            '\u00ad': '', '\ufeff': '', '\u200b': '', '\u200c': '', '\u200d': '',
        }
        for char, replacement in unicode_map.items():
            code = code.replace(char, replacement)

        # ── 2. Add axes.flatten() after plt.subplots ────────────────────────
        lines = code.splitlines()
        result_lines = []
        for line in lines:
            result_lines.append(line)
            if ('plt.subplots(' in line and
                ('fig' in line or 'axes' in line) and
                line.count('(') == line.count(')')):
                result_lines.append(
                    'axes = axes.flatten() if hasattr(axes, "flatten") else [axes]'
                )
        code = '\n'.join(result_lines)

        # ── 3. Remove duplicate matplotlib backend setup ────────────────────
        backend_count = code.count("matplotlib.use('Agg')")
        if backend_count > 1:
            code = re.sub(
                r'import matplotlib\s*\nmatplotlib\.use\([\'"]Agg[\'"]\)\s*\n',
                '',
                code,
                count=backend_count - 1,
            )

        # ── 4. Wrap file reads in try/except ──────────────────────────────    ──
        lines = code.splitlines()
        new_lines = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            sp = ' ' * indent

            is_read_call = bool(
            re.match(r'\w+\s*=\s*pd\.read_(?:csv|excel|parquet|json|table)\s*\(', stripped) or
            re.match(r'\w+\s*=\s*joblib\.load\s*\(', stripped) or
            re.match(r'\w+\s*=\s*pickle\.load\s*\(', stripped)
        )

        # Check previous line safely using index, not lines.index(line)
        prev_line = lines[idx - 1] if idx > 0 else ''
        already_in_try = 'try:' in prev_line

        if is_read_call and not already_in_try:
            var_name = stripped.split('=')[0].strip()
            new_lines.append(f"{sp}try:")
            new_lines.append(f"{sp}    {stripped}")
            new_lines.append(f"{sp}except (FileNotFoundError, OSError, Exception) as _read_err:")
            new_lines.append(f"{sp}    print(f'WARNING: Could not load file: {{_read_err}}')")
            new_lines.append(f"{sp}    {var_name} = None")
        else:
            new_lines.append(line)
        code = '\n'.join(new_lines)

        # ── 5. Replace raise statements with graceful exits ──────────────────
        code = re.sub(
            r'raise\s+FileNotFoundError\s*\(([^)]*)\)',
            lambda m: f"print(f'WARNING: File not found - {m.group(1)}'); import sys; sys.exit(0)",
            code,
        )
        code = re.sub(
            r'raise\s+SystemExit\s*\(([^)]*)\)',
            lambda m: "import sys; sys.exit(0)",
            code,
        )

        # ── 6. Fix assert statements → if/print ─────────────────────────────
        code = re.sub(
            r'assert\s+(.+?),\s*["\'](.+?)["\'](?:\s*$)',
            lambda m: f'if not ({m.group(1).strip()}):\n    print(f"WARNING: {m.group(2).strip()}")',
            code,
            flags=re.MULTILINE,
        )
        code = re.sub(
            r'assert\s+(.+?)(?:\s*$)',
            lambda m: f'if not ({m.group(1).strip()}):\n    print("WARNING: assertion failed")',
            code,
            flags=re.MULTILINE,
        )

        return code
    
    # ADD this new method to CodeExecutionAgent class, after _fix_common_code_errors:

    def _validate_and_fix_syntax(self, code: str) -> tuple[str, bool]:
        """Validate Python syntax. If broken, use the LLM to fix it."""
        import ast

        def _check(c: str) -> tuple[bool, str]:
            try:
                ast.parse(c)
                return True, ""
            except SyntaxError as e:
                return False, f"Line {e.lineno}: {e.msg}\n{e.text}"

        valid, error = _check(code)
        if valid:
            return code, True

        logger.warning(f"[CodeAgent] Syntax error: {error} — asking LLM to fix")

        fix_prompt = f"""This Python code has a syntax error. Fix ONLY the syntax error and return the complete corrected script.

Syntax error:
{error}

Code with error:
```python
{code}
```

Return the complete fixed Python script in ```python ... ``` fences.
Do not change logic, only fix the syntax."""

        try:
            from multimodal_ds.core.llm_client import chat_with_fallback
            fixed_raw = chat_with_fallback(
                primary_model=CODER_MODEL,
                fallback_model="ollama/qwen2.5:7b",
                messages=[
                    {"role": "system", "content": "You are a Python syntax fixer. Fix syntax errors only. Return complete code in ```python``` fences."},
                    {"role": "user", "content": fix_prompt},
                ],
                max_tokens=8000,
                temperature=0.0,
            )
            fixed_code = self._extract_code(fixed_raw)
            if fixed_code:
                valid, error = _check(fixed_code)
                if valid:
                    logger.info("[CodeAgent] LLM self-healed the syntax error")
                    return fixed_code, True
                else:
                    logger.warning(f"[CodeAgent] LLM fix still has syntax error: {error}")
        except Exception as e:
            logger.warning(f"[CodeAgent] LLM syntax fix failed: {e}")

        return code, False

    def _fix_filename_references(self, code: str, working_files: list[str]) -> str:
        """Detect and fix filename mismatches in generated code.

        The LLM often generates code with wrong filenames (e.g., 'Churn.csv' when
        the actual file is 'Churn_Modelling.csv'). This method:
        1. Finds file references in generated code (pd.read_csv, pd.read_excel, etc.)
        2. Checks if referenced files exist in working directory
        3. If not, finds the closest matching actual file and replaces the reference
        """
        if not working_files:
            return code

        actual_files = {Path(f).name.lower(): Path(f).name for f in working_files}
        import re

        # Patterns that reference data files in code
        file_ref_patterns = [
            r'pd\.read_(?:csv|excel|parquet|json)\s*\(\s*["\']([^"\']+)["\']',
            r'read_(?:csv|excel|parquet|json)\s*\(\s*["\']([^"\']+)["\']',
            r'open\s*\(\s*["\']([^"\']+\.(?:csv|json|txt))["\']',
        ]

        def find_closest_match(missing_name: str) -> str | None:
            """Find the most similar actual file using fuzzy matching."""
            import os
            missing_lower = missing_name.lower()
            # Exact match (case-insensitive)
            if missing_lower in actual_files:
                return actual_files[missing_lower]

            # Extract stem (filename without extension) for smarter matching
            missing_stem = os.path.splitext(missing_name)[0].lower()
            missing_ext = os.path.splitext(missing_name)[1].lower()

            # Try to find a match by comparing stems
            for actual_lower, actual_name in actual_files.items():
                actual_stem = os.path.splitext(actual_lower)[0]
                actual_ext = os.path.splitext(actual_lower)[1]

                # Skip if extensions don't match (could be different file types)
                if actual_ext != missing_ext:
                    continue

                # Check if stems share significant overlap:
                # 1. One stem is contained in the other (e.g., "Churn" in "Churn_Modelling")
                # 2. They share the first half of characters (e.g., "custom" vs "customer")
                if (missing_stem in actual_stem or actual_stem in missing_stem or
                    (len(missing_stem) > 3 and len(actual_stem) > 3 and
                     missing_stem[:len(missing_stem)//2] in actual_stem)):
                    return actual_name

            return None

        # Find all file references and try to fix them
        for pattern in file_ref_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for ref in matches:
                ref_lower = ref.lower()
                # Check if this exact reference exists
                if ref_lower not in actual_files:
                    # Try to find a close match
                    matched = find_closest_match(ref)
                    if matched:
                        logger.info(f"[CodeAgent] Fixing filename: '{ref}' -> '{matched}'")
                        # Replace all occurrences of this wrong filename using a compiled regex
                        old_ref_pattern = re.compile(
                            r'(["\'])' + re.escape(ref) + r'(["\'])',
                            re.IGNORECASE
                        )
                        code = old_ref_pattern.sub(
                            lambda m, matched=matched: m.group(1) + matched + m.group(2),
                            code
                        )

        # If no CSV filename is referenced, add a fallback import
        if not re.search(r"\.csv", code):
            csv_files = [Path(f).name for f in working_files if f.lower().endswith('.csv')]
            if csv_files:
                actual_filename = csv_files[0]
                fallback_line = f"# Data file: {actual_filename}\ndf = pd.read_csv('{actual_filename}')"
                # Insert after matplotlib import lines if they exist (first three lines usually)
                lines = code.splitlines()
                insert_idx = 0
                # Detect the typical three matplotlib lines
                if len(lines) >= 3 and all('matplotlib' in lines[i] for i in range(3)):
                    insert_idx = 3
                else:
                    # Find first line containing 'import matplotlib'
                    for i, l in enumerate(lines):
                        if 'import matplotlib' in l:
                            insert_idx = i + 1
                            break
                # Insert the fallback line
                lines = lines[:insert_idx] + [fallback_line] + lines[insert_idx:]
                code = "\n".join(lines)
        return code

    def execute_task(self, task: dict, data_context: str = "", file_paths: Optional[list] = None, max_retries: int = 2) -> dict:
        task_desc = task.get("description", str(task))
        task_name = task.get("name", "task")
        logger.info(f"[CodeAgent] Executing: {task_name}")

        with agent_span(self.AGENT_NAME, self.session_id, self._tracker) as span:
            span.set_metadata({"task_name": task_name})
            past_context = self._get_relevant_memory(task_desc)
            raw_code = self._generate_code(task_desc, data_context, past_context)
            code = self._extract_code(raw_code)
            if not code:
                # Retry code generation once with a simpler prompt before giving up.
                # The first attempt may fail if the model is still loading or
                # the context is too long. A simplified retry often succeeds.
                logger.warning("[CodeAgent] First code generation attempt returned empty — retrying with simplified prompt")
                simplified_desc = (
                    f"{task_desc}\n\n"
                    f"IMPORTANT: Respond with ONLY a Python code block inside ```python ... ``` fences. "
                    f"No explanation. No prose. Just the code."
                )
                raw_code = self._generate_code(simplified_desc, data_context[:500], "")
                code = self._extract_code(raw_code)
            if not code:
                logger.error(f"[CodeAgent] Code generation failed after retry for task: {task_desc[:100]}")
                return {
                    "success": False,
                    "error": "Code generation failed",
                    "code": "",
                    "output": "Code generation failed — LLM returned no parseable Python code.",
                    "files_created": [],
                }
            span.set_chars(input_chars=len(task_desc) + len(data_context), output_chars=len(code))
            result = self._execute_with_retry(code, task_desc, data_context, file_paths, max_retries)
            span.set_metadata({"task_name": task_name, "success": result["success"], "files_created": result["files_created"]})

        status_msg = "successfully" if result["success"] else "with errors"
        self.memory.store_analysis_step(
            step_name=task_name,
            result=f"Code executed {status_msg}.\nOutput: {result['output'][:500]}\nFiles: {result['files_created']}",
            session_id=self.session_id,
        )
        return result

    def execute(self, task_description: str, data_context: str = "", file_paths: Optional[list] = None, max_retries: int = 2) -> dict:
        rag_context = self._retrieve_rag_context(task_description)
        if rag_context:
            data_context = f"Relevant document context (from ChromaDB):\n{rag_context}\n\n" + data_context
        if file_paths:
            file_list = "\n".join(f"  - {Path(fp).name}" for fp in file_paths)
            data_context = f"Available data files (use exact names):\n{file_list}\n\n{data_context}"
        task = {"name": task_description[:80], "description": task_description}
        return self.execute_task(task=task, data_context=data_context, file_paths=file_paths, max_retries=max_retries)

    def _retrieve_rag_context(self, query: str, k: int = 4) -> str:
        try:
            results = self.memory.retrieve(query, n_results=k)
            if results:
                return "\n\n".join(r["content"] for r in results if r.get("content"))
        except Exception:
            pass
        return ""

    def _generate_code(self, task_desc: str, data_context: str, past_context: str) -> str:
        from multimodal_ds.core.llm_client import chat_with_fallback

        prompt = f"""Task: {task_desc}\nData Context:\n{data_context[:1500]}\nPrevious Context:\n{past_context[:500]}\nWorking directory: {self.working_dir}\nWrite Python code. Save all outputs to the current directory."""

        # Use unified LLM client - handles opencode/ and ollama/ prefixes automatically
        try:
            result = chat_with_fallback(
                primary_model=CODER_MODEL,
                fallback_model="ollama/qwen2.5:7b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=6000,
                temperature=0.1,
            )
            if result and not result.startswith("[Error:"):
                # Return raw LLM response; extraction will be performed later
                return result.strip()
            logger.warning(f"[CodeAgent] LLM returned error: {result}")
        except Exception as e:
            logger.error(f"[CodeAgent] Code generation failed: {e}")
        logger.warning(f"[CodeAgent] Returning empty code for task — LLM call failed or response unparseable")
        return ""

    def _execute_code(self, code: str, file_paths: Optional[list] = None):
        # Log execution details for debugging
        logger.debug(f"[CodeAgent] Preparing to execute script in {self.working_dir} (script will be written to temporary file)")
        # Log first three lines of the generated script to verify backend setup
        script_preview = "\n".join(code.splitlines()[:3])
        logger.debug(f"[CodeAgent] Script preview (first 3 lines):\n{script_preview}")
        files_before = set(self.working_dir.glob("*"))
        script_path = None
        copied_files = []

        # Copy data files to working dir so code can find them locally
        # Use a dedicated temp subdir inside working_dir (same filesystem → fast rename)
        if file_paths:
            for fp in file_paths:
                src = Path(fp)
                if src.exists():
                    dst = self.working_dir / src.name
                    if not dst.exists():
                        try:
                            # Try hard-link first (instant, zero copy) — works when
                            # src and dst are on the same filesystem
                            try:
                                os.link(src, dst)
                            except (OSError, NotImplementedError):
                                # Cross-filesystem or unsupported — fall back to copy
                                shutil.copy2(src, dst)
                            copied_files.append(dst)
                        except Exception as e:
                            logger.warning(f"[CodeAgent] Failed to copy {src.name}: {e}")

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=self.working_dir, delete=False, encoding="utf-8") as f:
                f.write(code)
                script_path = Path(f.name)
            
            import os as _os_exec
            _exec_env = _os_exec.environ.copy()
            _exec_env["PYTHONIOENCODING"] = "utf-8:replace"
            _exec_env["PYTHONUTF8"] = "1"

            run_kwargs = {
                "args": [sys.executable, "-X", "utf8", str(script_path)],
                "cwd": str(self.working_dir),
                "capture_output": True,
                "text": True,
                "timeout": _PROC_TIMEOUT_S,
                "env": _exec_env,
                "encoding": "utf-8",
                "errors": "replace",
            }
            if sys.platform != "win32":
                run_kwargs["preexec_fn"] = _sandbox_preexec
            
            result = subprocess.run(**run_kwargs)
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            # Log stderr on error for debugging
            if result.returncode != 0:
                logger.error(f"[CodeAgent] Subprocess exited with code {result.returncode}. Stderr (first 500 chars): {stderr[:500]}")
            # Always include full stderr for debugging, truncate stdout separately
            stderr_section = f"\n[stderr]:\n{stderr}" if stderr else ""
            combined = stdout[:_STDOUT_CHARS] + stderr_section
            
            success = result.returncode == 0 or (result.returncode != 0 and _is_only_warnings(stderr))
            
            if not success:
                logger.error(
                    f"[CodeAgent] Subprocess exited with code {result.returncode}.\n"
                    f"FULL STDERR:\n{stderr}\n"
                    f"STDOUT (first 3000):\n{stdout[:3000]}"
                )
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {_PROC_TIMEOUT_S}s", []
        except Exception as e:
            return False, f"Execution error: {e}", []
        finally:
            if script_path and script_path.exists():
                try: script_path.unlink()
                except Exception: pass
            # Cleanup copied data files to keep sandbox clean
            for cf in copied_files:
                try: cf.unlink()
                except Exception: pass

        files_after = set(self.working_dir.glob("*"))
        new_files = [f.name for f in (files_after - files_before) if f.is_file() and f.suffix != ".py"]
        return success, combined, new_files

    def _execute_with_retry(
        self,
        code: str,
        task_desc: str,
        data_context: str,
        file_paths: Optional[list],
        max_retries: int,
    ) -> dict:
        """
        Self-healing execution loop.
        On every failure: the LLM analyzes the error, fixes the code,
        and the sanitizer cleans it mechanically before re-execution.
        Builds an error history so each retry benefits from all prior failures.
        """
        import re as _re

        # Step 1: Ensure all imports are installed
        self._ensure_dependencies(code)

        # Step 2: Fix filenames
        if file_paths:
            working_files = [str(Path(fp).name) for fp in file_paths]
            code = self._fix_filename_references(code, working_files)

        # Step 3: Apply mechanical sanitizer
        code = self._fix_common_code_errors(code)

        # Step 4: Inject UTF-8 + warnings preamble
        PREAMBLE = (
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "import os as _os_preamble\n"
            "_os_preamble.environ['PYTHONIOENCODING'] = 'utf-8'\n"
            "import sys as _sys_preamble\n"
            "_sys_preamble.stdout.reconfigure(encoding='utf-8', errors='replace') "
            "if hasattr(_sys_preamble.stdout, 'reconfigure') else None\n\n"
        )
        # Strip ALL leading matplotlib/warnings/os/sys preamble lines the LLM added
        # so we don't duplicate them from PREAMBLE
        preamble_patterns = [
            r'^import matplotlib\b.*\n',
            r'^matplotlib\.use\([\'"]Agg[\'"]\).*\n',
            r'^import matplotlib\.pyplot.*\n',
            r'^import warnings\b.*\n',
            r'^warnings\.filterwarnings.*\n',
            r'^import os as _os_preamble.*\n',
            r'^import sys as _sys_preamble.*\n',
            r'^_os_preamble\.environ.*\n',
            r'^_sys_preamble\.stdout.*\n',
        ]
        for pat in preamble_patterns:
            code = _re.sub(pat, '', code, flags=_re.MULTILINE)

        # Hard-strip main() patterns before prepending preamble
        code = _strip_main_wrapper(code)

        code = PREAMBLE + code.lstrip('\n')

        # Step 5: Validate and LLM-fix syntax before first execution
        code, syntax_ok = self._validate_and_fix_syntax(code)
        if not syntax_ok:
            return {
                "success": False, "code": code,
                "output": "Unfixable syntax error in generated code.",
                "files_created": [], "error": "Syntax error", "retries_used": 0,
            }

        # Step 6: Execute with full self-healing retry loop
        error_history = []  # Accumulates all prior errors for increasingly informed fixes

        try:
            success, output, files = self._execute_code(code, file_paths)
        except Exception as e:
            success, output, files = False, str(e), []

        if success:
            return {
                "success": True, "code": code, "output": output,
                "files_created": files, "error": "", "retries_used": 0,
            }

        error_history.append(output)

        for attempt in range(max_retries):
            logger.info(f"[CodeAgent] Self-healing attempt {attempt + 1}/{max_retries}")

            # Build cumulative error context — each retry knows about ALL prior failures
            cumulative_error_context = "\n\n".join([
                f"--- Attempt {i+1} error ---\n{err[:1000]}"
                for i, err in enumerate(error_history)
            ])

            try:
                fix_code = self._generate_fix(
                    failed_code=code,
                    error_output=cumulative_error_context,
                    task_desc=task_desc,
                )
            except Exception as e:
                logger.warning(f"[CodeAgent] Fix generation raised: {e}")
                break

            if not fix_code:
                logger.warning(f"[CodeAgent] No fix generated on attempt {attempt + 1}")
                continue

            # Apply full pipeline to fixed code
            self._ensure_dependencies(fix_code)
            if file_paths:
                fix_code = self._fix_filename_references(fix_code, working_files)
            fix_code = self._fix_common_code_errors(fix_code)
            for pat in preamble_patterns:
                fix_code = _re.sub(pat, '', fix_code, flags=_re.MULTILINE)
            fix_code = _strip_main_wrapper(fix_code)
            fix_code = PREAMBLE + fix_code.lstrip('\n')
            fix_code, syntax_ok = self._validate_and_fix_syntax(fix_code)

            if not syntax_ok:
                logger.warning(f"[CodeAgent] Fixed code still has syntax errors on attempt {attempt + 1}")
                error_history.append("Syntax error in LLM-generated fix — could not parse.")
                continue

            try:
                success, output, files = self._execute_code(fix_code, file_paths)
            except Exception as e:
                success, output, files = False, str(e), []

            if success:
                return {
                    "success": True, "code": fix_code, "output": output,
                    "files_created": files, "error": "", "retries_used": attempt + 1,
                }

            error_history.append(output)
            code = fix_code  # Next attempt fixes the most recent version

        return {
            "success": False, "code": code, "output": output,
            "files_created": files, "error": output, "retries_used": max_retries,
        }

    def _generate_fix(self, failed_code: str, error_output: str, task_desc: str) -> str:
        """
        LLM-powered self-healer. Provides complete environment context so the
        LLM can fix ANY error without hardcoded rules.
        """
        from multimodal_ds.core.llm_client import chat_with_fallback
        import sys
        import subprocess

        # Introspect the actual environment — no hardcoding
        def _get_installed_packages() -> str:
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'list', '--format=columns'],
                    capture_output=True, text=True, timeout=15,
                )
                return result.stdout[:3000]
            except Exception:
                return "Could not retrieve installed packages"

        env_info = (
            f"Python: {sys.version}\n"
            f"Platform: {sys.platform}\n"
            f"Default encoding: {sys.getdefaultencoding()}\n"
            f"Stdout encoding: {getattr(sys.stdout, 'encoding', 'unknown')}\n"
            f"Working directory contents: {[f.name for f in self.working_dir.iterdir() if f.is_file()]}\n"
            f"Pandas version: {self._get_pkg_version('pandas')}\n"
            f"Sklearn version: {self._get_pkg_version('sklearn')}\n"
            f"Numpy version: {self._get_pkg_version('numpy')}\n"
            f"Matplotlib version: {self._get_pkg_version('matplotlib')}\n"
        )

        installed = _get_installed_packages()

        prompt = f"""You are an expert Python debugger with 50 years of data science experience.
Fix this Python script that failed. Use ONLY packages from the installed list below.

=== ENVIRONMENT ===
{env_info}

=== INSTALLED PACKAGES (use ONLY these) ===
{installed}

=== TASK ===
{task_desc}

=== ERROR(S) — ALL PRIOR ATTEMPTS ===
{error_output[:4000]}

=== FAILED CODE ===
```python
{failed_code[-5000:]}
```

=== SELF-HEALING RULES ===
1. Analyze the EXACT error message — fix the ROOT CAUSE, not symptoms
2. If ImportError: check installed packages list above — use an alternative that IS installed
3. If FileNotFoundError: add Path existence check, use graceful fallback
4. If UnicodeEncodeError: replace all non-ASCII chars with ASCII equivalents
5. If KeyError on DataFrame column: always verify column exists with `if col in df.columns`
6. If encoding issues: add sys.stdout.reconfigure(encoding='utf-8') at top
7. If model.pkl not found in eval/viz tasks: train a quick fallback model inline
8. NEVER use packages not in the installed list above
9. ALWAYS wrap file reads in try/except
10. Return a COMPLETE, RUNNABLE script — not just the fix

Return the complete fixed Python script in ```python ... ``` fences."""

        try:
            result = chat_with_fallback(
                primary_model=self._get_coder_model(),
                fallback_model="ollama/qwen2.5:7b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert Python debugger. "
                            "Analyze errors carefully and fix root causes. "
                            "Return ONLY the complete fixed script in ```python``` fences. "
                            "Never explain — just fix and return code."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8000,
                temperature=0.0,
            )
            if result and not result.startswith("[Error:"):
                return self._extract_code(result)
        except Exception as e:
            logger.error(f"[CodeAgent] Fix generation failed: {e}")
        return ""

    def _get_coder_model(self) -> str:
        """Return the configured coder model."""
        from multimodal_ds.config import CODER_MODEL
        return CODER_MODEL


    def _get_pkg_version(self, pkg: str) -> str:
        try:
            import importlib.metadata
            return importlib.metadata.version(pkg)
        except Exception:
            return "unknown"

    def _extract_code(self, text: str) -> str:
        import re

        if not text or not text.strip():
            return ""

        # Strip <think>...</think> reasoning blocks (qwen3, deepseek-r1)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Try ```python ... ``` fence first (allow optional spaces, optional space after backticks, and Windows line endings)
        m = re.search(r'```\s*python\s*\r?\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            if code:
                return code

        # Try any ``` ... ``` fence (any language tag or none)
        m = re.search(r'```.*?\r?\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            if code and any(
                kw in code
                for kw in ('import ', 'from ', 'def ', 'class ', '#', 'pd.', 'df', 'print(')
            ):
                return code

        # Last resort: raw text that looks like Python code. Never return fenced text
        # by filtering out lines that start with markdown code block fences (```).
        cleaned_lines = []
        for line in text.splitlines():
            stripped_line = line.strip()
            if stripped_line.startswith("```"):
                continue
            cleaned_lines.append(line)
        cleaned_text = "\n".join(cleaned_lines).strip()

        if cleaned_text:
            first_line = cleaned_text.split('\n')[0].strip()
            if any(first_line.startswith(kw) for kw in (
                'import ', 'from ', 'def ', 'class ', '#', 'pd.', 'df', 'print('
            )):
                return cleaned_text

        logger.warning("[CodeAgent] Could not extract Python code from LLM response")
        logger.debug(f"[CodeAgent] Raw response (first 300 chars): {text[:300]!r}")
        return ""

    def _get_relevant_memory(self, query: str) -> str:
        memories = self.memory.retrieve(query, n_results=3)
        if not memories:
            return ""
        return "\n".join(m["content"][:200] for m in memories)
