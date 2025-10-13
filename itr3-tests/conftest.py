import sys, os, pytest

# --- ensure FastAPI app is importable ---
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# --- mock required env vars ---
os.environ["GCV_API_KEY"] = "fake-key-for-tests"

import main  # import after fixing path


@pytest.fixture(autouse=True)
def override_auth_dependency():
    """Force all routes to bypass BasicAuth for tests."""
    main.app.dependency_overrides[main.get_current_username] = lambda: "test_user"
    yield
    main.app.dependency_overrides.clear()
