import pytest
from multimodal_ds.core.message_bus import reset_bus


@pytest.fixture(autouse=True, scope="function")
def reset_message_bus():
    """Reset the global MessageBus singleton before AND after every test.

    Original fixture was scope='session' — the bus was reset once at the
    start of the test session but never between individual tests. Handlers
    subscribed in test_evaluation_agent.py (EVAL_REQUEST, EVAL_FLAGGED,
    EVAL_COMPLETE) remained active when test_visualization_agent.py ran,
    causing those handlers to receive VIZ_REQUEST messages they weren't
    designed for and appending spurious entries to their received[] lists.

    scope='function' + autouse=True guarantees every test starts with a
    clean bus regardless of execution order or parallelism.

    The reset runs BEFORE the test (via the initial reset_bus() call) and
    AFTER the test (via the yield + finally block) so teardown is also clean
    even if the test raises an exception.
    """
    reset_bus()
    yield
    reset_bus()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary output directory for tests that write files."""
    return tmp_path


@pytest.fixture
def tmp_output_dir(temp_output_dir):
    """Provide the same temporary output directory fixture name expected by older tests."""
    return temp_output_dir

