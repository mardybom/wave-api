import pytest
import dyslexia_myths


class FakeCur:
    def __init__(self, rows=None):
        self.rows = rows
    def execute(self, *a, **k): pass
    def fetchall(self): return self.rows
    def __enter__(self): return self
    def __exit__(self, *a): pass


class FakeConn:
    def __init__(self, rows=None):
        self.rows = rows
    def cursor(self, **kwargs):  # accept cursor_factory argument
        return FakeCur(self.rows)
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass



def test_fetch_next_myth_row_success(monkeypatch):
    monkeypatch.setattr(
        dyslexia_myths,
        "get_db_connection",
        lambda: FakeConn([{"id": 1, "myth": "m", "truth": "t"}])  # added id
    )
    result = dyslexia_myths.fetch_next_myth_row()
    assert result[0]["myth"] == "m"



def test_fetch_next_myth_row_none(monkeypatch):
    monkeypatch.setattr(
        dyslexia_myths,
        "get_db_connection",
        lambda: FakeConn([])
    )
    result = dyslexia_myths.fetch_next_myth_row()
    assert result == []