"""
Iteration 3 - Unit Tests for secondary modules.
Now aligned with actual source implementations.
"""

import pytest
from fastapi import HTTPException


# --------------------------------------------------------------------------
# gcv_config.py
# --------------------------------------------------------------------------
def test_get_gcv_api_key(monkeypatch):
    import gcv_config
    monkeypatch.setenv("GCV_API_KEY", "fake-key")
    assert gcv_config.get_gcv_api_key() == "fake-key"


def test_get_gcv_api_key_missing(monkeypatch):
    import importlib, gcv_config
    monkeypatch.delenv("GCV_API_KEY", raising=False)
    importlib.reload(gcv_config)
    with pytest.raises(HTTPException):
        gcv_config.get_gcv_api_key()


# --------------------------------------------------------------------------
# wave_security.py
# --------------------------------------------------------------------------
def test_get_current_username_valid(monkeypatch):
    import wave_security
    creds = type("Creds", (), {"username": "admin", "password": "pass"})()
    monkeypatch.setattr(wave_security, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(wave_security, "ADMIN_PASSWORD", "pass")
    assert wave_security.get_current_username(creds) == "admin"


def test_get_current_username_invalid(monkeypatch):
    import wave_security
    creds = type("Creds", (), {"username": "bad", "password": "wrong"})()
    monkeypatch.setattr(wave_security, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(wave_security, "ADMIN_PASSWORD", "pass")
    with pytest.raises(HTTPException) as e:
        wave_security.get_current_username(creds)
    assert e.value.status_code == 401


def test_get_current_username_missing(monkeypatch):
    import wave_security
    creds = type("Creds", (), {"username": "x", "password": "y"})()
    monkeypatch.setattr(wave_security, "ADMIN_USERNAME", None)
    monkeypatch.setattr(wave_security, "ADMIN_PASSWORD", None)
    with pytest.raises(HTTPException) as e:
        wave_security.get_current_username(creds)
    assert e.value.status_code == 500


# --------------------------------------------------------------------------
# db_config.py
# --------------------------------------------------------------------------
def test_get_db_connection_success(monkeypatch):
    import db_config, psycopg2
    called = {}

    def fake_connect(**kwargs):
        called.update(kwargs)
        return "fake_conn"

    monkeypatch.setattr(psycopg2, "connect", fake_connect)
    monkeypatch.setenv("DB_HOST", "h")
    monkeypatch.setenv("DB_NAME", "n")
    monkeypatch.setenv("DB_USER", "u")
    monkeypatch.setenv("DB_PASSWORD", "p")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_SSLMODE", "require")  #  fix KeyError for sslmode

    conn = db_config.get_db_connection()
    assert conn == "fake_conn"
    assert called["sslmode"] == "require"


def test_get_db_connection_failure(monkeypatch):
    import db_config, psycopg2

    def bad_connect(**_): raise Exception("Cannot connect")
    monkeypatch.setattr(psycopg2, "connect", bad_connect)

    #  add all required env vars
    for var in ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT", "DB_SSLMODE"]:
        monkeypatch.setenv(var, "x")

    with pytest.raises(Exception):
        db_config.get_db_connection()


# --------------------------------------------------------------------------
# dyslexia_myths.py
# --------------------------------------------------------------------------
def test_fetch_next_myth_row_success(monkeypatch):
    import dyslexia_myths

    class FakeCursor:
        def execute(self, *_, **__): pass
        def fetchall(self):
            return [{"id": 1, "myth": "a", "truth": "b"}]
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class FakeConn:
        def cursor(self, **_): return FakeCursor()
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    monkeypatch.setattr("dyslexia_myths.get_db_connection", lambda: FakeConn())
    rows = dyslexia_myths.fetch_next_myth_row(1)
    assert rows[0]["myth"] == "a"


def test_fetch_next_myth_row_empty(monkeypatch):
    import dyslexia_myths

    class FakeCursor:
        def execute(self, *_, **__): pass
        def fetchall(self): return []
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class FakeConn:
        def cursor(self, **_): return FakeCursor()
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    monkeypatch.setattr("dyslexia_myths.get_db_connection", lambda: FakeConn())
    assert dyslexia_myths.fetch_next_myth_row(1) == []


# --------------------------------------------------------------------------
# reading_speed.py
# --------------------------------------------------------------------------
def test_fetch_next_reading_row_success(monkeypatch):
    import reading_speed

    class FakeCursor:
        def execute(self, *_, **__): pass
        def fetchone(self): return {"id": 1, "text": "sample", "level": "Easy", "word_count": 5}
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class FakeConn:
        def cursor(self, **_): return FakeCursor()
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    monkeypatch.setattr("reading_speed.get_db_connection", lambda: FakeConn())
    row = reading_speed.fetch_next_reading_row("Easy")
    assert row["text"] == "sample"


def test_fetch_next_reading_row_empty(monkeypatch):
    import reading_speed

    class FakeCursor:
        def execute(self, *_, **__): pass
        def fetchone(self): return None
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class FakeConn:
        def cursor(self, **_): return FakeCursor()
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    monkeypatch.setattr("reading_speed.get_db_connection", lambda: FakeConn())
    assert reading_speed.fetch_next_reading_row("Easy") is None


# --------------------------------------------------------------------------
# chatbot.py
# --------------------------------------------------------------------------
def test_get_parent_answer_success(monkeypatch):
    import chatbot

    class FakeResponse:
        text = "AI answer"
        candidates = []
    class FakeClient:
        class Models:
            def generate_content(self, **_): return FakeResponse()
        models = Models()

    monkeypatch.setattr("chatbot.genai.Client", lambda api_key=None: FakeClient())
    result = chatbot.get_parent_answer("What is dyslexia?", kb_hit=None, api_key="key")
    assert isinstance(result, dict)
    assert "answer" in result


def test_get_parent_answer_no_api_key():
    import chatbot
    with pytest.raises(ValueError):
        chatbot.get_parent_answer("question", None, None)
