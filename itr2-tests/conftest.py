import os
import pytest
import sys
import base64

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("DB_HOST", "fake_host")
    monkeypatch.setenv("DB_NAME", "fake_db")
    monkeypatch.setenv("DB_USER", "fake_user")
    monkeypatch.setenv("DB_PASSWORD", "fake_pass")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_SSLMODE", "require")
    monkeypatch.setenv("GCV_API_KEY", "fake_api_key")

@pytest.fixture
def valid_b64() -> str:
    # any valid base64
    return base64.b64encode(b"hello").decode("utf-8")

@pytest.fixture
def build_vision_response():
    """
    Build a minimal Vision DOCUMENT_TEXT_DETECTION-like JSON.
    Usage: build_vision_response([("A", 0.9), ("B", 0.8)])
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