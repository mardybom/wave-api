import pytest
import sentence_rearranging as mod


def test_fetch_next_sentence_row(monkeypatch):
    fake_row = {
        "sentence_id": 5,
        "original_sentence": "Hello world",
        "jumbled_sentence": "world Hello",
        "difficulty_level": "easy",
    }

    class FakeCursor:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def execute(self, sql, params=None): self.sql = sql
        def fetchone(self): return fake_row

    class FakeConn:
        def cursor(self, *a, **k): return FakeCursor()
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    monkeypatch.setattr(mod, "get_db_connection", lambda: FakeConn())
    row = mod.fetch_next_sentence_row("easy")
    assert row == fake_row
    assert row["sentence_id"] == 5

def test_no_rows_rolls_back_and_returns_none(monkeypatch):
    # Capture the FakeConn instance to assert rollback was called
    holder = {}

    class FakeCursor:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def execute(self, sql, params=None): pass
        def fetchone(self):
            # no rows → triggers rollback + return None
            return None

    class FakeConn:
        def __init__(self):
            self.rollback_called = False
            self.commit_called = False
            self.closed = False
        def cursor(self, *a, **k): return FakeCursor()
        def commit(self): self.commit_called = True
        def rollback(self): self.rollback_called = True
        def close(self): self.closed = True

    def fake_get_conn():
        c = FakeConn()
        holder["conn"] = c
        return c

    monkeypatch.setattr(mod, "get_db_connection", fake_get_conn)

    out = mod.fetch_next_sentence_row("easy")
    assert out is None
    conn = holder["conn"]
    assert conn.rollback_called is True
    assert conn.commit_called is False
    assert conn.closed is True

def test_exception_path_rolls_back_and_raises(monkeypatch):
    holder = {}

    class ExplodingCursor:
        def __init__(self):
            self.calls = 0
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def execute(self, sql, params=None):
            self.calls += 1
            # after taking advisory lock, raise on the next execute
            if self.calls >= 2:
                raise RuntimeError("db explode")
        def fetchone(self):
            # would be used if no explosion
            return {"sentence_id": 1, "original_sentence": "a", "jumbled_sentence": "a", "difficulty_level": "easy"}

    class FakeConn:
        def __init__(self):
            self.rollback_called = False
            self.closed = False
        def cursor(self, *a, **k): return ExplodingCursor()
        def commit(self): pass
        def rollback(self): self.rollback_called = True
        def close(self): self.closed = True

    def fake_get_conn():
        c = FakeConn()
        holder["conn"] = c
        return c

    monkeypatch.setattr(mod, "get_db_connection", fake_get_conn)

    with pytest.raises(RuntimeError):
        mod.fetch_next_sentence_row("easy")

    conn = holder["conn"]
    assert conn.rollback_called is True
    assert conn.closed is True