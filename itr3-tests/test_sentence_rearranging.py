import pytest
import sentence_rearranging


class FakeCursor:
    def __init__(self, row=None): self._row = row
    def execute(self, *a, **k): pass
    def fetchone(self): return self._row
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


class FakeConn:
    def __init__(self, row=None, fail=False):
        self.row = row
        self.fail = fail
    def cursor(self, *a, **k):
        if self.fail: raise Exception("DB fail")
        return FakeCursor(self.row)
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass


def test_fetch_success(monkeypatch):
    monkeypatch.setattr(
        sentence_rearranging,
        "get_db_connection",
        lambda: FakeConn({"sentence_id": 1, "original_sentence": "A", "jumbled_sentence": "B"})
    )
    result = sentence_rearranging.fetch_next_sentence_row("easy")
    assert result["original_sentence"] == "A"


def test_fetch_none(monkeypatch):
    monkeypatch.setattr(sentence_rearranging, "get_db_connection", lambda: FakeConn(None))
    result = sentence_rearranging.fetch_next_sentence_row("easy")
    assert result is None


def test_fetch_db_error(monkeypatch):
    monkeypatch.setattr(sentence_rearranging, "get_db_connection", lambda: FakeConn(fail=True))
    with pytest.raises(Exception):
        sentence_rearranging.fetch_next_sentence_row("easy")