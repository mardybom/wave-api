import pytest
import image_labeling


def test_format_label():
    """Ensure label is cleaned properly."""
    assert image_labeling.format_label("cat_dog-1!") == "cat dog 1"  # ✅ fixed expected


def test_generate_rearranged_labels():
    """Ensure rearranged labels differ from original."""
    results = image_labeling.generate_rearranged_labels("abcd", count=3)
    assert all(r != "abcd" for r in results)


def test_label_to_arpabet(monkeypatch):
    """Map label to ARPAbet phoneme."""
    monkeypatch.setattr("pronouncing.phones_for_word", lambda _: ["T EH S T"])
    result = image_labeling.label_to_arpabet("test")
    assert "T EH S T" in result


def test_label_to_arpabet_none(monkeypatch):
    """Return empty string if no phoneme found."""
    monkeypatch.setattr("pronouncing.phones_for_word", lambda _: [])
    assert image_labeling.label_to_arpabet("xyz") == ""


def test_fetch_random_image_row_success(monkeypatch):
    """Simulate valid DB row."""
    fake_row = {"image_id": 1, "image_byte": b"bytes", "image_label": "dog"}

    class FakeCur:
        def execute(self, *a, **k): pass
        def fetchone(self): return fake_row
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class FakeConn:
        def cursor(self): return FakeCur()
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    monkeypatch.setattr(image_labeling, "get_db_connection", lambda: FakeConn())
    monkeypatch.setattr(image_labeling, "label_to_arpabet", lambda _: "ARP")

    result = image_labeling.fetch_random_image_row()
    assert result["image_label"] == "dog"


def test_fetch_random_image_row_none(monkeypatch):
    """Return None when no image found."""
    class FakeCur:
        def execute(self, *a, **k): pass
        def fetchone(self): return None
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class FakeConn:
        def cursor(self): return FakeCur()
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    monkeypatch.setattr(image_labeling, "get_db_connection", lambda: FakeConn())
    assert image_labeling.fetch_random_image_row() is None
