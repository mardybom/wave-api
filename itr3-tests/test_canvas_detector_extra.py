import pytest
import canvas_detector
from fastapi import HTTPException


def test_level_letter_mismatch_hard_case():
    """Hard level expects 2 letters, single should raise 400."""
    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", "A", "capital", "hard")
    assert e.value.status_code == 400
    assert "must be 2 letters" in e.value.detail


def test_no_text_in_both_passes(monkeypatch):
    """Vision API returns no results in both TEXT and DOCUMENT detection."""
    calls = []

    def fake_post(*a, **k):
        class FakeResp:
            def raise_for_status(self): pass
            def json(self):
                calls.append(1)
                return {"responses": [{}]}  # no text found
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)

    result = canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "fake-key", "A", "capital", "easy")
    assert isinstance(result, dict)
    assert "letters" in result
    assert result["detected_count"] == 0


def test_vision_api_timeout(monkeypatch):
    """Simulate network timeout causing unexpected error."""
    def fake_post(*a, **k):
        raise Exception("Timeout while calling Vision API")

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "fake-key", "A", "capital", "easy")
    assert e.value.status_code == 500
    assert "Unexpected Vision API error" in e.value.detail


def test_partial_text_annotations(monkeypatch):
    """Vision API returns partial textAnnotations (simulating a noisy response)."""
    def fake_post(*a, **k):
        class FakeResp:
            def raise_for_status(self): pass
            def json(self):
                return {"responses": [{"textAnnotations": [{"description": " A "}, {"description": " "}]}]}
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    result = canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", "A", "capital", "easy")
    assert isinstance(result, dict)
    assert "expected_letter" in result


def test_api_returns_error_field(monkeypatch):
    """Simulate Vision API error field directly in responses."""
    def fake_post(*a, **k):
        class R:
            def raise_for_status(self): pass
            def json(self):
                return {"responses": [{"error": {"message": "Fake internal error"}}]}
        return R()

    monkeypatch.setattr("requests.post", fake_post)
    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", "A", "capital", "easy")
    assert e.value.status_code == 502


def test_fallback_text_not_found(monkeypatch):
    """Simulate Vision API returning both passes empty but valid format."""
    def fake_post(*a, **k):
        class R:
            def raise_for_status(self): pass
            def json(self): return {"responses": [{}]}
        return R()

    monkeypatch.setattr("requests.post", fake_post)
    result = canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "fake-key", "A", "capital", "easy")
    assert "detected_count" in result
    assert result["detected_count"] == 0

# ---- Non-alphabet expected_letter ---- #
def test_invalid_expected_letter_type():
    """Non-alphabet letter should raise 400."""
    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", "12", "capital", "easy")
    assert e.value.status_code == 400
    assert "Aa-Zz" in e.value.detail


# ---- Level mismatch (easy vs hard logic) ---- #
@pytest.mark.parametrize("expected,level", [("AB", "easy"), ("A", "hard")])
def test_invalid_level_combination(expected, level):
    """Letter count not matching level should raise validation error."""
    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", expected, "capital", level)
    assert e.value.status_code == 400
    assert "must be" in e.value.detail


# ---- Case mismatch (capital vs small) ---- #
@pytest.mark.parametrize("letter,flag", [("a", "capital"), ("A", "small")])
def test_capitalization_mismatch(letter, flag):
    """Mismatch between flag and case raises 400."""
    with pytest.raises(HTTPException):
        canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", letter, flag, "easy")


# ---- Vision API raises internal Exception ---- #
def test_unexpected_vision_exception(monkeypatch):
    """Any exception during Vision call should yield 500."""
    def fake_post(*a, **k): raise Exception("Vision internal crash")
    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", "A", "capital", "easy")
    assert e.value.status_code == 500
    assert "Unexpected Vision API error" in e.value.detail


# ----  Vision API 502 error (response.error field) ---- #
def test_vision_error_field(monkeypatch):
    """Simulate Vision API JSON error response."""
    def fake_post(*a, **k):
        class FakeResp:
            def raise_for_status(self): pass
            def json(self):
                return {"responses": [{"error": {"message": "Fake Vision fail"}}]}
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", "A", "capital", "easy")
    assert e.value.status_code == 502
    assert "Fake Vision fail" in e.value.detail


# ---- Fallback (first TEXT_DETECTION empty, second works) ---- #
def test_fallback_document_text_detection(monkeypatch):
    """First call returns no textAnnotations, second returns valid."""
    calls = []

    def fake_post(*a, **k):
        class R:
            def raise_for_status(self): pass
            def json(self):
                if not calls:
                    calls.append(1)
                    return {"responses": [{}]}  # empty first call
                else:
                    return {"responses": [{"textAnnotations": [{"description": "A"}]}]}
        return R()

    monkeypatch.setattr("requests.post", fake_post)
    result = canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "fake-key", "A", "capital", "easy")
    assert isinstance(result, dict)
    assert "expected_letter" in result


# ---- Empty Vision responses (no crash) ---- #
def test_empty_vision_responses(monkeypatch):
    """If Vision API returns completely empty, function raises 500 as unexpected error."""
    def fake_post(*a, **k):
        class R:
            def raise_for_status(self): pass
            def json(self): return {"responses": []}
        return R()

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(HTTPException) as e:
        canvas_detector.detect_handwritten_letters_from_base64("ZmFrZQ==", "key", "A", "capital", "easy")
    assert e.value.status_code == 500
    assert "Unexpected Vision API error" in e.value.detail