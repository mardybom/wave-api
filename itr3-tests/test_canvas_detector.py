from unittest import result
import pytest
from fastapi import HTTPException
import canvas_detector


def test_detect_valid(monkeypatch):
    """Simulate a valid Vision API response."""
    def fake_post(*args, **kwargs):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"responses": [{"fullTextAnnotation": {"text": "A"}}]}  #  match Vision API response shape
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

def test_invalid_expected_letter_non_alpha():
    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64(
            "ZmFrZQ==", "key", "A1", "capital", "easy"
        )
    assert e.value.status_code == 400


def test_is_capital_mismatch():
    with pytest.raises(HTTPException):
        canvas_detector.detect_handwritten_letters_from_base64(
            "ZmFrZQ==", "key", "a", "capital", "easy"
        )


def test_level_mismatch_easy_too_long():
    with pytest.raises(HTTPException):
        canvas_detector.detect_handwritten_letters_from_base64(
            "ZmFrZQ==", "key", "AB", "capital", "easy"
        )


def test_level_mismatch_hard_too_short():
    with pytest.raises(HTTPException):
        canvas_detector.detect_handwritten_letters_from_base64(
            "ZmFrZQ==", "key", "A", "capital", "hard"
        )

# --- Vision API response edge cases ---

def test_invalid_base64(monkeypatch):
    """Trigger invalid base64 decoding path (75)."""
    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64(
            "!!invalid!!", "key", "A", "capital", "easy"
        )
    assert "Invalid base64" in e.value.detail


def test_empty_responses(monkeypatch):
    """Covers 179–184, empty list returned."""
    def fake_post(*a, **k):
        class R:
            def raise_for_status(self): pass
            def json(self): return {"responses": []}
        return R()
    monkeypatch.setattr("requests.post", fake_post)
    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64(
            "ZmFrZQ==", "key", "A", "capital", "easy"
        )
    assert e.value.status_code == 500


def test_request_exception(monkeypatch):
    """Covers 152–154, simulate requests error."""
    def bad_post(*a, **k):
        raise Exception("Connection failed")
    monkeypatch.setattr("requests.post", bad_post)
    with pytest.raises(HTTPException):
        canvas_detector.detect_handwritten_letters_from_base64(
            "ZmFrZQ==", "key", "A", "capital", "easy"
        )


def test_unexpected_exception(monkeypatch):
    """Covers 199–200 unexpected top-level error."""
    def bad_post(*a, **k):
        raise ValueError("Random failure")
    monkeypatch.setattr("requests.post", bad_post)
    with pytest.raises(HTTPException):
        canvas_detector.detect_handwritten_letters_from_base64(
            "ZmFrZQ==", "key", "A", "capital", "easy"
        )

def test_error_key_in_response(monkeypatch):
    """Covers Vision API 'error' block (lines 132–140)."""
    def fake_post(*a, **k):
        class R:
            def raise_for_status(self): pass
            def json(self): return {"responses": [{"error": {"message": "Bad"}}]}
        return R()

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64(
            "ZmFrZQ==", "key", "A", "capital", "easy"
        )
    assert e.value.status_code == 502
    # Match the real detail string
    assert "Bad" in e.value.detail