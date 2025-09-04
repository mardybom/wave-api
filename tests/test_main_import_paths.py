import sys
import importlib

def test_cors_alt_branch_import(monkeypatch):
    
    # Force non-wildcard origins so the other branch executes on import
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:5500")

    # Ensure a clean re-import so module-level code runs again
    sys.modules.pop("main", None)
    import main  # noqa: F401

    # Optional: sanity check the app exists
    assert hasattr(main, "app")