"""
Parent Chatbot Module
Provides parent-friendly answers about dyslexia using Gemini with Google Search grounding.
"""

from google import genai
from google.genai import types


def get_parent_answer(question: str, kb_hit: str | None = None, api_key: str | None = None) -> dict:
    """
    Generate a grounded, parent-friendly response about dyslexia.

    Args:
        question (str): Parent's question text.
        kb_hit (Optional[str]): Optional knowledge base context.

    Returns:
        dict: Structured chatbot response with answer, sources, and suggestions.
    """
    if not api_key:
        raise ValueError("API key is required for get_parent_answer()")
    
    client = genai.Client(api_key=api_key)

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool], temperature=0.7)

    system_prompt = """
        You are 'Parent Help' — a warm, factual assistant for parents of school-aged children with dyslexia.
        Your role:
        - Give clear, evidence-based information about dyslexia.
        - Be friendly, empathetic, and easy to understand.
        - Never diagnose or give medical/treatment advice.
        Tone:
        - Write like a supportive teacher.
        - Use short, clear sentences (2–4 max).
        - Be kind and reassuring, not clinical.
        ---
        Rules:
        1. **Greetings / Thanks**
        If user greets or thanks you → reply warmly and invite dyslexia-related questions.
        2. **Dyslexia Questions**
        - Give concise, factual, parent-friendly answers.
        - Mention trusted sources (e.g., Australian Dyslexia Association) when relevant.
        3. **Treatment Questions**
        - Reply only: “Sorry, I’m not supposed to provide medical suggestions. Please seek advice from a registered psychologist.”
        4. **Unrelated Topics**
        - Reply only: “Sorry, I can only answer questions about dyslexia.”
        6. **Clarification / Small Talk**
        - “What can you do?” → “I can answer general questions about dyslexia and learning support.”
        - “Who are you?” → “I’m Parent Help, a friendly assistant for parents learning about dyslexia.”
        - Unclear input → “Could you rephrase your question about dyslexia?”
        7. **Distress or Emotional Tone**
        - Respond empathetically and suggest professional support:
            “I’m sorry you’re feeling that way. If you need to talk to someone, contact Lifeline (13 11 14) in Australia.”
        Format for informative answers:
        Answer: <Main answer>
        1. <Point 1> — <Summary> — <Citation (if any)>
        2. <Point 2> — <Summary> — <Citation (if any)>
        """

    prompt_text = (
        f"{system_prompt}\n\n"
        f"Question: {question}\n"
        f"Context from our knowledge base:\n{kb_hit or 'N/A'}"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_text,
        config=config,
    )

    answer = response.text.strip() if hasattr(response, "text") else str(response)
    candidate = response.candidates[0] if getattr(response, "candidates", None) else None

    sources = _extract_sources(candidate) if candidate else []
    suggestions = _extract_suggestions(candidate) if candidate else []

    return {
        "answer": answer,
        "sources": sources,
        "suggestions": suggestions,
        "disclaimer": "This is general information, not medical advice.",
    }


def _extract_sources(candidate) -> list[str]:
    """Extract grounded web or retrieved sources."""
    links = set()
    gm = getattr(candidate, "grounding_metadata", None)
    if not gm:
        return []

    for chunk in getattr(gm, "grounding_chunks", []) or []:
        for attr in ("web", "retrieved_context"):
            src = getattr(chunk, attr, None)
            if src:
                uri = getattr(src, "uri", "") or getattr(src, "url", "") or getattr(src, "source_uri", "")
                title = getattr(src, "title", "") or "Source"
                if uri:
                    links.add((title, uri))

    return [f"[{t}]({u})" for t, u in links]


def _extract_suggestions(candidate) -> list[str]:
    """Extract related search suggestions if available."""
    gm = getattr(candidate, "grounding_metadata", None)
    if not gm:
        return []

    se = getattr(gm, "search_entry_point", None)
    if not se:
        return []

    for attr in ("suggested_queries", "suggestedQuestions", "suggestions"):
        maybe = getattr(se, attr, None)
        if isinstance(maybe, (list, tuple)):
            return [str(x) for x in maybe]
    return []
