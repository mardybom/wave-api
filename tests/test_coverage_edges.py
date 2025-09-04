import pytest
from fastapi import HTTPException

def test_canvas_detector_malformed_json_triggers_generic_500(monkeypatch, valid_b64):
    """
    Force resp.json() to run with an unexpected ValueError.
    This bypasses your handled branches and lands in the outer
    'except Exception as e' => HTTPException(500, "Error processing image: ...")
    """
    from canvas_detector import detect_handwritten_letters_from_base64

    class Resp:
        status_code = 200
        text = "ok"
        def json(self):
            raise ValueError("malformed")

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    with pytest.raises(HTTPException) as ei:
        detect_handwritten_letters_from_base64(valid_b64, "key", "A")
    assert ei.value.status_code == 500
    assert "Error processing image: malformed" in ei.value.detail


def test_api_http_exception_passthrough(client, monkeypatch):
    """
    Ensure the 'except HTTPException: raise' line in main.py is executed.
    We patch the detector (as imported by main) to raise a 418.
    """
    import main
    monkeypatch.setattr(
        main, "detect_handwritten_letters_from_base64",
        lambda *args, **kwargs: (_ for _ in ()).throw(HTTPException(status_code=418, detail="teapot"))
    )

    resp = client.post("/alphabet_mastery", json={"canvas_input": "aGVsbG8=", "expected_letter": "A"})
    assert resp.status_code == 418
    assert "teapot" in resp.text