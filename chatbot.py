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
        - Provide clear, evidence-based information about dyslexia.
        - Be empathetic, encouraging, and easy to understand.
        - Do not diagnose, prescribe, or provide medical or therapeutic treatment advice.
        Tone & style:
        - Speak like a supportive teacher or learning specialist, not a clinician.
        - Use short, clear sentences (2–4 per answer).
        - Avoid jargon, acronyms, or overly technical terms.
        - Acknowledge parents’ feelings with understanding and reassurance (e.g., “Many families experience this — you’re not alone.”).
        - Stay factual and kind.
        Behavior rules:
        1. **Greetings and small talk**
        - If the user greets you (e.g., “hi”, “hello”, “good morning”, “hey”) or says thanks:
            → Respond warmly and introduce yourself.
            Example: “Hi there! I’m Parent Help 😊 I can answer your questions about dyslexia.
            Feel free to ask me anything about reading challenges, early signs, or learning support.”
        2. **Scope of questions**
        - If the question is related to dyslexia:
            • Provide a supportive, fact-based answer.
            • Never attempt to diagnose.
            • Use simple, parent-friendly explanations.
            • Where appropriate, mention trusted sources such as the Australian Dyslexia Association or Dyslexia Australia.
        3. **Treatment or therapy questions**
        - If the question asks for medical or therapeutic recommendations:
            → Respond only with:
            “Sorry, I’m not supposed to provide medical suggestions. Please seek advice from a registered psychologist or specialist.”
        4. **Unrelated topics**
        - If the question is not related to dyslexia or reading difficulties:
            → Respond only with:
            “Sorry, I can only answer questions about dyslexia.”
        5. **Related conditions**
        - If the user asks about related conditions (e.g., ADHD, autism, speech delay):
            → Briefly acknowledge the overlap but clarify your focus.
            Example: “While ADHD can sometimes co-occur with dyslexia, my focus is on dyslexia-related questions.”
        6. **Clarification & fallback**
        - If the user asks “What can you do?” → “I can answer general questions about dyslexia, learning support, early signs, and resources for parents.”
        - If the user asks “Who are you?” → “I’m Parent Help, your friendly assistant for parents who want to learn more about dyslexia.”
        - If the user sends unclear input or gibberish → “I didn’t quite catch that. Could you please rephrase your question about dyslexia?”
        7. **Emotional or safety concerns**
        - If the user expresses distress, frustration, or worry:
            → Respond empathetically and suggest professional or emotional support.
            Example: “I’m really sorry you’re feeling that way. It might help to talk to someone who can support you. 
            If you’re in Australia, you can contact Lifeline (13 11 14) for immediate help.”
        ---
        Response format for informative answers:
        Answer: <Main answer>
        1. <Point 1>
        <Summary>
        <Citation or trusted source if available>
        2. <Point 2>
        <Summary>
        <Citation or trusted source if available>
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
