import base64
import pytest
import image_labeling

def test_format_label():
    # Double spaces are expected in current implementation when multiple underscores exist
    assert image_labeling.format_label("hello_world-123") == "hello world 123"
    assert image_labeling.format_label("   messy!!__label ") == "messy  label"

def test_generate_rearranged_labels():
    fake_labels = image_labeling.generate_rearranged_labels("cake", count=4)
    assert all(l != "cake" for l in fake_labels)
    assert len(fake_labels) <= 4
    assert len(set(fake_labels)) == len(fake_labels)

def test_fetch_random_image_row(monkeypatch):
    class FakeCursor:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def execute(self, sql, params=None): pass
        def fetchone(self):
            return {
                "image_id": 1,
                "image_label": "cake",
                "image_byte": b"fakebytes"
            }

    class FakeConn:
        def cursor(self, *a, **k): return FakeCursor()
        def close(self): pass

    monkeypatch.setattr(image_labeling, "get_db_connection", lambda: FakeConn())
    row = image_labeling.fetch_random_image_row()
    assert row["image_id"] == 1
    assert row["image_label"] == "cake"
    assert isinstance(row["image_base64"], str)
    assert "cake" in row["options"]

def test_generate_rearranged_labels_single_char():
    # 'a' → only one permutation; after discarding original, nothing left
    out = image_labeling.generate_rearranged_labels("a", count=4)
    assert out == []