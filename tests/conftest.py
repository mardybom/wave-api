import os
import sys
import base64
import pytest
from fastapi.testclient import TestClient

# make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

@pytest.fixture(scope="session")
def app():
    from main import app as fastapi_app
    return fastapi_app

@pytest.fixture(scope="session")
def client(app):
    return TestClient(app)

@pytest.fixture(autouse=True)
def ensure_env(monkeypatch):
    
    # Your endpoint checks this env var:
    monkeypatch.setenv("GCV_API_KEY", "test-key")
    yield

@pytest.fixture
def valid_b64() -> str:
    
    # Any valid base64 is fine (your detector only validates base64)
    return base64.b64encode(b"hello").decode("utf-8")

@pytest.fixture
def build_vision_response():
    """
    Build a minimal Google Vision-like JSON for DOCUMENT_TEXT_DETECTION.
    Use like: payload = build_vision_response([("A",0.91),("B",0.4),("a",0.88)])
    """
    def _make(letters):
        symbols = [{"text": ch, "confidence": conf} for ch, conf in letters]
        return {
            "responses": [{
                "fullTextAnnotation": {
                    "pages": [{
                        "blocks": [{
                            "paragraphs": [{
                                "words": [{
                                    "symbols": symbols
                                }]
                            }]
                        }]
                    }]
                }
            }]
        }
    return _make