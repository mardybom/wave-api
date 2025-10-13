import pytest
import chatbot

# --- Mock Gemini Client for testing ---
class DummyResponse:
    def __init__(self, text="AI reply"):
        self.text = text


class DummyModel:
    def __init__(self, should_fail=False, text="AI reply"):
        self.should_fail = should_fail
        self.text = text

    def generate_content(self, *a, **k):
        if self.should_fail:
            raise Exception("Gemini internal failure")
        return DummyResponse(self.text)


class DummyClient:
    def __init__(self, model=None):
        self.models = model or DummyModel()


# --- Tests ---

def test_empty_query(monkeypatch):
    """Empty question should still return a structured dict."""
    monkeypatch.setattr("chatbot.genai.Client", lambda api_key=None: DummyClient())
    result = chatbot.get_parent_answer("", "context", "fake-key")
    assert isinstance(result, dict)
    assert "answer" in result


def test_empty_api_key(monkeypatch):
    """Empty API key should raise ValueError."""
    monkeypatch.setattr("chatbot.genai.Client", lambda api_key=None: DummyClient())
    with pytest.raises(ValueError):
        chatbot.get_parent_answer("What is dyslexia?", "context", "")


def test_missing_text_field(monkeypatch):
    """Gemini returns an object without 'text' -> still returns answer dict."""
    class BrokenResponse:
        pass

    class BrokenModel:
        def generate_content(self, *a, **k):
            return BrokenResponse()

    monkeypatch.setattr("chatbot.genai.Client", lambda api_key=None: DummyClient(BrokenModel()))
    result = chatbot.get_parent_answer("Hi", "kb", "fake-key")
    assert isinstance(result, dict)
    assert "answer" in result  # ✅ match actual function behavior


def test_gemini_internal_failure(monkeypatch):
    """Gemini throws exception -> function should propagate error."""
    monkeypatch.setattr(
        "chatbot.genai.Client",
        lambda api_key=None: DummyClient(DummyModel(should_fail=True))
    )
    with pytest.raises(Exception) as excinfo:
        chatbot.get_parent_answer("Hi", "kb", "fake-key")
    assert "Gemini internal failure" in str(excinfo.value)
