import psycopg2
import pytest
from db import _conninfo, get_db_connection

def test_conninfo(mock_env):
    info = _conninfo()
    assert info["host"] == "fake_host"
    assert info["dbname"] == "fake_db"
    assert info["user"] == "fake_user"
    assert info["password"] == "fake_pass"
    assert info["port"] == "5432"
    assert info["sslmode"] == "require"

def test_get_db_connection(monkeypatch):
    called = {}

    def fake_connect(**kwargs):
        called.update(kwargs)
        return "fake_connection"

    monkeypatch.setattr(psycopg2, "connect", fake_connect)
    conn = get_db_connection()
    assert conn == "fake_connection"
    assert called["dbname"] == "fake_db"
