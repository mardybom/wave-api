import pytest
import reading_speed


class FakeCur:
    def __init__(self, row=None): self.row = row
    def execute(self, *a, **k): pass
    def fetchone(self): return self.row
    def __enter__(self): return self
    def __exit__(self, *a): pass


class FakeConn:
    def __init__(self, row=None):
        self.row = row
    def cursor(self, **kwargs):  #fix here
        return FakeCur(self.row)
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass



def test_fetch_next_reading_row_success(monkeypatch):
    """Return valid reading passage."""
    fake = {"id": 1, "text": "abc", "level": "Easy", "word_count": 3}  # correct keys
    monkeypatch.setattr(reading_speed, "get_db_connection", lambda: FakeConn(fake))
    result = reading_speed.fetch_next_reading_row("Easy")
    assert result["text"] == "abc"



def test_fetch_next_reading_row_none(monkeypatch):
    """Return None when no passage found."""
    monkeypatch.setattr(reading_speed, "get_db_connection", lambda: FakeConn(None))
    result = reading_speed.fetch_next_reading_row("Easy")
    assert result is None