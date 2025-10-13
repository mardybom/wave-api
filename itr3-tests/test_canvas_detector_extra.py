import pytest
import base64
import canvas_detector
from fastapi import HTTPException


def test_invalid_expected_letter_non_alpha():
    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", "123", "capital", "easy")
    assert e.value.status_code == 400
    assert "Aa-Zz" in e.value.detail


def test_case_mismatch_between_flag_and_letter():
    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", "a", "capital", "easy")
    assert e.value.status_code == 400


def test_invalid_level_combination():
    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", "AB", "capital", "easy")
    assert "must be 1 letter" in e.value.detail


def test_empty_base64_image():
    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("", "key", "A", "capital", "easy")
    assert "Empty base64" in e.value.detail


def test_invalid_base64_image():
    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("!!!", "key", "A", "capital", "easy")
    assert e.value.status_code == 400


def test_vision_api_returns_error(monkeypatch):
    def fake_post(*a, **k):
        class FakeR:
            def raise_for_status(self): pass
            def json(self):
                return {"responses": [{"error": {"message": "API failed"}}]}
        return FakeR()
    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", "A", "capital", "easy")
    assert e.value.status_code == 502


def test_vision_api_fallback(monkeypatch):
    """Simulate first TEXT_DETECTION empty then fallback DOCUMENT_TEXT_DETECTION."""
    calls = []

    def fake_post(*a, **k):
        class FakeR:
            def raise_for_status(self): pass
            def json(self):
                # First call returns no textAnnotations; second call returns result
                if not calls:
                    calls.append(1)
                    return {"responses": [{}]}
                else:
                    return {"responses": [{"textAnnotations": [{"description": "A"}]}]}
        return FakeR()

    monkeypatch.setattr("requests.post", fake_post)
    result = canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", "A", "capital", "easy")
    assert isinstance(result, dict)
    assert "expected_letter" in result