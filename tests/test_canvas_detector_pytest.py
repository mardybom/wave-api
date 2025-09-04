import pytest
from fastapi import HTTPException

def test_expected_letter_missing(valid_b64):
    from canvas_detector import detect_handwritten_letters_from_base64
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", None)
    assert ei.value.status_code == 400
    assert "expected_letter is required" in ei.value.detail

@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_base64(bad):
    from canvas_detector import detect_handwritten_letters_from_base64
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(bad, "key", "A")
    assert ei.value.status_code == 400
    assert "Empty base64 image" in ei.value.detail


def test_invalid_base64():
    from canvas_detector import detect_handwritten_letters_from_base64
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64("!!!notb64!!!", "key", "A")
    assert ei.value.status_code == 400
    assert "Invalid base64 image" in ei.value.detail


@pytest.mark.parametrize("bad", ["AB", "1", "-", "Aa", "AA", " "])
def test_expected_letter_invalid(valid_b64, bad):
    from canvas_detector import detect_handwritten_letters_from_base64
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", bad)
    assert ei.value.status_code == 400
    assert "expected_letter must be a single A–Z letter" in ei.value.detail


def test_vision_non_200(monkeypatch, valid_b64):
    from canvas_detector import detect_handwritten_letters_from_base64
    class Resp:
        status_code = 403
        text = "Forbidden"
        def json(self): return {}
    def fake_post(*a, **k): return Resp()
    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", "A")
    assert ei.value.status_code == 502
    assert "HTTP 403" in ei.value.detail

def test_vision_error_field(monkeypatch, valid_b64):
    from canvas_detector import detect_handwritten_letters_from_base64
    class Resp:
        status_code = 200
        def json(self): return {"responses": [{"error": {"message": "quota exceeded"}}]}
        text = "ok"
    def fake_post(*a, **k): return Resp()
    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", "A")
    assert ei.value.status_code == 502
    assert "quota exceeded" in ei.value.detail


def test_request_exception(monkeypatch, valid_b64):
    from canvas_detector import detect_handwritten_letters_from_base64
    import requests
    def boom(*a, **k):
        raise requests.exceptions.Timeout("too slow")
    monkeypatch.setattr(requests, "post", boom)

    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", "A")
    assert ei.value.status_code == 502
    assert "Vision API request failed" in ei.value.detail


def test_success_match(monkeypatch, valid_b64, build_vision_response):
    from canvas_detector import detect_handwritten_letters_from_base64
    # Letters: A with high confidence twice, one mismatch 'B'
    letters = [("A", 0.91), ("B", 0.40), ("a", 0.88)]
    payload_json = build_vision_response(letters)

    class Resp:
        status_code = 200
        text = "ok"
        def json(self): return payload_json
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    out = detect_handwritten_letters_from_base64(valid_b64, "key", "a")
    assert out["expected_letter"] == "A"
    assert out["detected_count"] == 3
    assert out["match_count"] == 2
    assert out["top_match_confidence"] == 0.91
    assert out["match_ratio"] == pytest.approx(2/3, rel=1e-3)
    assert out["is_correct"] is True
    assert out["reason"] == "match found with sufficient confidence"
    assert any(m["letter"] == "B" and m["count"] == 1 for m in out["mismatches"])


def test_success_but_low_confidence_and_ratio(monkeypatch, valid_b64, build_vision_response):
    from canvas_detector import detect_handwritten_letters_from_base64
    # Only one expected letter with low confidence; ratio = 1/2 = 0.5 (< 0.60), conf 0.5 (< 0.70) => False
    letters = [("A", 0.50), ("C", 0.90)]
    payload_json = build_vision_response(letters)

    class Resp:
        status_code = 200
        text = "ok"
        def json(self): return payload_json
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    out = detect_handwritten_letters_from_base64(valid_b64, "key", "A")
    assert out["is_correct"] is False
    assert "low confidence" in out["reason"]

def test_data_url_base64(monkeypatch, build_vision_response):
    from canvas_detector import detect_handwritten_letters_from_base64
    # Minimal valid data URL
    data_url = "data:image/png;base64," + "aGVsbG8="  # "hello"

    payload_json = build_vision_response([("A", 0.95)])
    class Resp:
        status_code = 200
        text = "ok"
        def json(self): return payload_json

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    out = detect_handwritten_letters_from_base64(data_url, "key", "A")
    assert out["is_correct"] is True
    assert out["match_count"] == 1
    assert out["reason"] == "match found with sufficient confidence"

def test_no_letters_detected(monkeypatch, valid_b64):
    from canvas_detector import detect_handwritten_letters_from_base64

    class Resp:
        status_code = 200
        text = "ok"
        def json(self):
            return {"responses": [{"fullTextAnnotation": {"pages": []}}]}

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    out = detect_handwritten_letters_from_base64(valid_b64, "key", "A")
    assert out["detected_count"] == 0
    assert out["match_count"] == 0
    assert out["is_correct"] is False
    assert out["reason"] == "no matching letter detected"