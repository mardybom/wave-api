import pytest
from fastapi.testclient import TestClient
import main
from fastapi import HTTPException

client = TestClient(main.app)

def test_alphabet_mastery_success(monkeypatch):
    def fake_detect(canvas_input, api_key, expected_letter, is_capital, level):
        return {
            "expected_letter": expected_letter,
            "is_correct": True,
            "reason": "Match found with sufficient confident.",
            "detected_count": 3,
            "match_count": 3,
            "top_match_confidence": 0.9,
            "match_ratio": 1.0,
            "letters": [{"letter": expected_letter, "confidence": 0.9}],
            "mismatches": []
        }

    monkeypatch.setattr(main, "detect_handwritten_letters_from_base64", fake_detect)

    payload = {
        "canvas_input": "fake_base64",
        "expected_letter": "A",
        "is_capital": "capital",   # <-- string, not boolean
        "level": "easy"            # <-- string
    }
    response = client.post("/alphabet_mastery", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["is_correct"] is True
    assert data["reason"] == "Match found with sufficient confident."


def test_alphabet_mastery_missing_key(monkeypatch):
    monkeypatch.delenv("GCV_API_KEY", raising=False)

    payload = {
        "canvas_input": "fake_base64",
        "expected_letter": "A",
        "is_capital": "capital",   # string
        "level": "easy"
    }
    response = client.post("/alphabet_mastery", json=payload)
    assert response.status_code == 500
    assert "Missing GCV_API_KEY" in response.text

def test_sentence_next_success(monkeypatch):
    fake_row = {"sentence_id": 1, "original_sentence": "Hi", "jumbled_sentence": "iH", "difficulty_level": "easy"}
    monkeypatch.setattr(main, "fetch_next_sentence_row", lambda level: fake_row)
    response = client.post("/sentence/next", json={"level": "easy"})
    assert response.status_code == 200
    assert response.json()["data"]["original_sentence"] == "Hi"

def test_sentence_next_not_found(monkeypatch):
    monkeypatch.setattr(main, "fetch_next_sentence_row", lambda level: None)
    response = client.post("/sentence/next", json={"level": "hard"})
    assert response.status_code == 404

def test_image_labeling_next_success(monkeypatch):
    fake_row = {"image_id": 1, "image_label": "cake", "image_base64": "xxx", "options": ["cake", "kace"]}
    monkeypatch.setattr(main, "fetch_random_image_row", lambda: fake_row)
    response = client.post("/image_labeling/next", json={})
    assert response.status_code == 200
    assert response.json()["data"]["image_label"] == "cake"

def test_image_labeling_next_not_found(monkeypatch):
    monkeypatch.setattr(main, "fetch_random_image_row", lambda: None)

    response = client.post("/image_labeling/next", json={})
    assert response.status_code == 500
    # The wrapped message preserves the original detail inside "Database error: ..."
    assert "Database error" in response.text
    assert "No rows found in image_labeling table" in response.text

def test_alphabet_http_exception_passthrough(monkeypatch):
    # exercise: 'except HTTPException: raise' branch
    def boom(*a, **k):
        raise HTTPException(status_code=418, detail="teapot")
    monkeypatch.setattr(main, "detect_handwritten_letters_from_base64", boom)

    payload = {
        "canvas_input": "aGVsbG8=",
        "expected_letter": "A",
        "is_capital": "capital",
        "level": "easy",
    }
    r = client.post("/alphabet_mastery", json=payload)
    assert r.status_code == 418
    assert "teapot" in r.text

def test_alphabet_generic_error_wrapped(monkeypatch):
    # exercise: generic exception → 500 "Unexpected error: ..."
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(main, "detect_handwritten_letters_from_base64", boom)

    payload = {
        "canvas_input": "aGVsbG8=",
        "expected_letter": "A",
        "is_capital": "capital",
        "level": "easy",
    }
    r = client.post("/alphabet_mastery", json=payload)
    assert r.status_code == 500
    assert "Unexpected error: kaboom" in r.text

def test_sentence_next_empty_level_400():
    r = client.post("/sentence/next", json={"level": ""})
    assert r.status_code == 400
    assert "level is required" in r.text

def test_sentence_next_keyerror_500(monkeypatch):
    def boom(level):
        raise KeyError("DB_HOST")
    monkeypatch.setattr(main, "fetch_next_sentence_row", boom)

    r = client.post("/sentence/next", json={"level": "easy"})
    assert r.status_code == 500
    assert "Missing DB config env var" in r.text

def test_image_labeling_db_exception_wrapped(monkeypatch):
    import main
    from fastapi.testclient import TestClient
    client = TestClient(main.app)

    def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(main, "fetch_random_image_row", boom)

    r = client.post("/image_labeling/next", json={})
    assert r.status_code == 500
    assert "Database error: db down" in r.text