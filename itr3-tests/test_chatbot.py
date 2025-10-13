import pytest
import chatbot

def test_get_parent_answer_success(monkeypatch):
    """Mock a successful Gemini API call."""

    class FakeResponse:
        def __init__(self): self.text = "AI: Hello!"

    class FakeModel:
        def generate_content(self, *a, **k): return FakeResponse()

    class FakeClient:
        models = FakeModel()

    # Patch constructor genai.Client to return FakeClient
    monkeypatch.setattr("chatbot.genai.Client", lambda api_key=None: FakeClient())

    result = chatbot.get_parent_answer("What is dyslexia?", "kb", "fake-key")
    assert "answer" in result
    assert "AI" in result["answer"]

def test_get_parent_answer_failure(monkeypatch):
    """Simulate Gemini API crash."""
    class FakeModel:
        def generate_content(self, *a, **k): raise Exception("Gemini fail")

    class FakeClient:
        models = FakeModel()

    monkeypatch.setattr("chatbot.genai.Client", lambda api_key=None: FakeClient())

    with pytest.raises(Exception) as exc_info:  # expect exception
        chatbot.get_parent_answer("Hi", "kb", "fake-key")
    assert "Gemini fail" in str(exc_info.value)


