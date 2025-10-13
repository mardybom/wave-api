from unittest import result
import pytest
from fastapi import HTTPException
import canvas_detector


def test_detect_valid(monkeypatch):
    """Simulate a valid Vision API response."""
    def fake_post(*args, **kwargs):
        class R:
            status_code = 200
            def raise_for_status(self): pass  # ✅ required
            def json(self):
                return {"responses": [{"fullTextAnnotation": {"text": "A"}}]}  # ✅ match Vision API response shape
        return R()

    monkeypatch.setattr("requests.post", fake_post)

    result = canvas_detector.detect_handwritten_letters_from_base64(
        "ZmFrZQ==", "fake-key", "A", "capital", "easy"
    )
    assert "expected_letter" in result
    assert isinstance(result, dict)


def test_invalid_expected_letter():
    """Reject invalid expected letter input."""
    with pytest.raises(HTTPException):
        canvas_detector.detect_handwritten_letters_from_base64(
            "ZmFrZQ==", "key", "12", "capital", "easy"
        )


def test_missing_expected_letter():
    """Reject missing expected letter."""
    with pytest.raises(HTTPException):
        canvas_detector.detect_handwritten_letters_from_base64(
            "ZmFrZQ==", "key", None, "capital", "easy"
        )


def test_api_failure(monkeypatch):
    """Simulate Vision API crash."""
    def bad_post(*args, **kwargs):
        raise Exception("Vision API fail")

    monkeypatch.setattr("requests.post", bad_post)
    with pytest.raises(HTTPException):
        canvas_detector.detect_handwritten_letters_from_base64(
            "ZmFrZQ==", "key", "A", "capital", "easy"
        )
