import pytest
from fastapi import HTTPException
from canvas_detector import detect_handwritten_letters_from_base64

def test_expected_letter_required(valid_b64):
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", None, "capital", "easy")
    assert ei.value.status_code == 400
    assert "expected_letter is required" in ei.value.detail

@pytest.mark.parametrize("bad", ["", "  "])
def test_empty_base64(bad):
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(bad, "key", "A", "capital", "easy")
    assert ei.value.status_code == 400
    assert "Empty base64 image" in ei.value.detail

def test_invalid_base64():
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64("!!!", "key", "A", "capital", "easy")
    assert ei.value.status_code == 400
    assert "Invalid base64 image" in ei.value.detail

def test_is_capital_mismatch(valid_b64):
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", "A", "small", "easy")
    assert ei.value.status_code == 400
    assert "does not match" in ei.value.detail

def test_level_validation(valid_b64):
    # easy level but 2 letters
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", "AB", "capital", "easy")
    assert "must be 1 letter" in ei.value.detail

    # hard level but 1 letter
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", "A", "capital", "hard")
    assert "must be 2 letters" in ei.value.detail

def test_vision_non_200(monkeypatch, valid_b64):
    class Resp:
        status_code = 403
        text = "Forbidden"
        def json(self): return {}
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", "A", "capital", "easy")
    assert ei.value.status_code == 502
    assert "HTTP 403" in ei.value.detail

def test_vision_error_field(monkeypatch, valid_b64):
    class Resp:
        status_code = 200
        def json(self): return {"responses": [{"error": {"message": "quota exceeded"}}]}
        text = "ok"
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", "A", "capital", "easy")
    assert ei.value.status_code == 502
    assert "quota exceeded" in ei.value.detail

def test_request_exception(monkeypatch, valid_b64):
    import requests
    def boom(*a, **k): raise requests.exceptions.Timeout("too slow")
    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", "A", "capital", "easy")
    assert ei.value.status_code == 502
    assert "Vision API request failed" in ei.value.detail

def test_success_match_simple(monkeypatch, valid_b64, build_vision_response):
    # expected_letter=A, is_capital=capital, one symbol "A" only → zero mismatches => is_correct True
    payload_json = build_vision_response([("A", 0.91)])

    class Resp:
        status_code = 200
        text = "ok"
        def json(self): return payload_json

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    out = detect_handwritten_letters_from_base64(valid_b64, "key", "A", "capital", "easy")
    assert out["expected_letter"] == "A"
    assert out["detected_count"] == 1
    assert out["match_count"] == 1
    assert out["top_match_confidence"] == 0.91
    assert out["match_ratio"] == 1.0
    assert out["is_correct"] is True
    assert out["reason"] == "Match found with sufficient confident."
    assert out["mismatches"] == []

def test_success_with_data_url(monkeypatch, build_vision_response):
    data_url = "data:image/png;base64,aGVsbG8="  # "hello"
    payload_json = build_vision_response([("A", 0.95)])

    class Resp:
        status_code = 200
        text = "ok"
        def json(self): return payload_json

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    out = detect_handwritten_letters_from_base64(data_url, "key", "A", "capital", "easy")
    assert out["is_correct"] is True
    assert out["match_count"] == 1

def test_generic_exception_path(monkeypatch, valid_b64):
    # Force resp.json() to raise → triggers outer 'except Exception' → 500
    class Resp:
        status_code = 200
        text = "ok"
        def json(self):
            raise ValueError("malformed")

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", "A", "capital", "easy")
    assert ei.value.status_code == 500
    assert "Error processing image: malformed" in ei.value.detail

def test_expected_letter_nonalpha(valid_b64):
    # line 20: expected_letter must be alphabetic
    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", "A1", "capital", "easy")
    assert ei.value.status_code == 400
    assert "Aa-Zz" in ei.value.detail

def test_special_letters_capitalization_capital(monkeypatch, valid_b64, build_vision_response):
    # lines 75–78: branch that forces case for letters in 'cxvusmwyzp' when is_capital="capital"
    payload_json = build_vision_response([("c", 0.8)])  # 'c' is in the special set
    class Resp:
        status_code = 200
        text = "ok"
        def json(self): return payload_json
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    out = detect_handwritten_letters_from_base64(valid_b64, "key", "C", "capital", "easy")
    # ensure letter got uppercased by the special branch
    assert any(l["letter"] == "C" for l in out["letters"])

def test_special_letters_capitalization_small(monkeypatch, valid_b64, build_vision_response):
    # lines 75–78: same branch, but is_capital="small" => downcase
    payload_json = build_vision_response([("C", 0.8)])  # upper, but should be forced to lower
    class Resp:
        status_code = 200
        text = "ok"
        def json(self): return payload_json
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    out = detect_handwritten_letters_from_base64(valid_b64, "key", "c", "small", "easy")
    assert any(l["letter"] == "c" for l in out["letters"])

def test_reason_mismatched_alphabets(monkeypatch, valid_b64, build_vision_response):
    # lines 99–100 + 116–117: mismatches > 0 => "You have X mismatched alphabets"
    # expected two letters (hard), but only one detected
    payload_json = build_vision_response([("A", 0.9)])
    class Resp:
        status_code = 200
        text = "ok"
        def json(self): return payload_json
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    out = detect_handwritten_letters_from_base64(valid_b64, "key", "AB", "capital", "hard")
    assert out["is_correct"] is False
    assert "mismatched alphabets" in out["reason"]

def test_reason_low_confidence(monkeypatch, valid_b64, build_vision_response):
    # lines 104–105: low confidence branch
    payload_json = build_vision_response([("A", 0.2)])  # < 0.30
    class Resp:
        status_code = 200
        text = "ok"
        def json(self): return payload_json
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    out = detect_handwritten_letters_from_base64(valid_b64, "key", "A", "capital", "easy")
    assert out["is_correct"] is False
    assert "Low confidence" in out["reason"]

def test_extraneous_letter_counts_as_mismatch(monkeypatch, valid_b64, build_vision_response):
    # Detected "X" but expected "A" → hits the "up not in expected_up" branch (lines 99–100)
    payload_json = build_vision_response([("X", 0.8)])

    class Resp:
        status_code = 200
        text = "ok"
        def json(self): return payload_json

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    out = detect_handwritten_letters_from_base64(valid_b64, "key", "A", "capital", "easy")
    assert out["is_correct"] is False
    # ensure "X" is recorded as a mismatch (coming from the first mismatch loop)
    assert any(m["letter"] == "X" for m in out["mismatches"])