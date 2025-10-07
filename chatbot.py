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

    system_prompt = (
        "You are 'Parent Help', a warm, factual assistant for parents of school-aged "
        "children with dyslexia.\n"
        "- If the question is related to dyslexia:\n"
        "  • Answer only dyslexia-related questions.\n"
        "  • Never diagnose.\n"
        "  • Be empathetic, concise, and encouraging.\n"
        "  • Always cite sources when available.\n"
        "- If the question is about treatment suggestions:\n"
        "  → Respond only with:\n"
        "    'Sorry, I'm not supposed to provide medical suggestions. Please seek advice from a registered psychologist.'\n"
        "- If the question is unrelated to dyslexia:\n"
        "  → Respond only with:\n"
        "    'Sorry, I can't answer this question.'\n"
        "- Follow this response format:\n"
        "    Answer: <Main answer>\n"
        "    1. <Point 1>\n"
        "       <Summary>\n"
        "       <Citation>\n"
        "    2. <Point 2>\n"
        "       <Summary>\n"
        "       <Citation>\n"
    )

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
