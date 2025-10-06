# --- target-skeleton-style setup ---
import os
from google import genai
from google.genai import types

# 1) Client
# client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "api key"))

api_key = os.getenv("GCV_API_KEY")
client = genai.Client(api_key=api_key)

# 2) Grounding tool (Google Search)
grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

# 3) Config (you can add temperature here if you like)
config = types.GenerateContentConfig(
    tools=[grounding_tool],
    temperature=0.7
)

def get_parent_answer(question: str, kb_hit: str | None = None):
    """
    Builds a grounded, parent-friendly answer.
    Assumes globals exist (per your target skeleton):
      - client = genai.Client(...)
      - grounding_tool = types.Tool(google_search=types.GoogleSearch())
      - config = types.GenerateContentConfig(tools=[grounding_tool], ...)
    """
    # ---- 1) Compose prompt (system guidance + user/context in one string) ----
    sys_prompt = f'''You are 'Parent Help', a warm, factual assistant for parents of school-aged 
    agent with dyslexia. Please answer the question following the criteria below.
    - If the question is related to kids with dyslexia:
        - Please answer the question following the criteria below.
            - Only answer dyslexia-related questions.
            - NEVER diagnose.
            - Be empathetic, concise and encouraging.
            - Always cite sources when available.
    - If question asked are related to treatment suggestions on dyslexia:
        - return this particular reponse and no other response:
            - Sorry I'm not suppose to provide medical suggestion. Please seek advice from a registered psychologist
    - If question asked are not related to dyslexia at all:
        - return this particular reponse and no other response:
            - Sorry I can't answer this question.
    - follow this exact format when returning the response, the <> is just a placeholder, you can replace it with your answer in the returned response
        Answer: <Main answer to question>
        1. <point 1>
        <summary on point1>
        <relevant citation to point1>
        2. <point 2>
        <summary on point2>
        <relevant citation to point2>
        '''
    prompt_text = (
        f"{sys_prompt}\n\n"
        f"Question: {question}\n"
        f"Context from our knowledge base:\n{kb_hit or 'N/A'}"
    )

    # ---- 2) Call Gemini with Google Search grounding (target skeleton style) ----
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_text,    # single string is fine with google-genai
        config=config            # must include tools=[grounding_tool]
    )

    # ---- 3) Extract main text ----
    answer = response.text.strip() if hasattr(response, "text") else str(response)

    # ---- 4) Extract grounded citations from grounding_chunks (defensive) ----
    def _extract_sources_from_candidate(cand) -> list[str]:
        links: set[tuple[str, str]] = set()
        gm = getattr(cand, "grounding_metadata", None)
        if not gm:
            return []
        chunks = getattr(gm, "grounding_chunks", []) or []
        for ch in chunks:
            # Case A: web source
            web = getattr(ch, "web", None)
            if web:
                uri = getattr(web, "uri", "") or getattr(web, "url", "")
                title = getattr(web, "title", "") or "Source"
                if uri:
                    links.add((title, uri))
            # Case B: retrieved_context (e.g., from other connectors)
            rc = getattr(ch, "retrieved_context", None)
            if rc:
                uri = getattr(rc, "uri", "") or getattr(rc, "source_uri", "")
                title = getattr(rc, "title", "") or "Source"
                if uri:
                    links.add((title, uri))
        return [f"[{t}]({u})" for (t, u) in links]

    candidate = response.candidates[0] if getattr(response, "candidates", None) else None
    sources = _extract_sources_from_candidate(candidate) if candidate else []

    # ---- 5) Extract search suggestions if exposed by this SDK build ----
    suggestions: list[str] = []
    if candidate:
        gm = getattr(candidate, "grounding_metadata", None)
        if gm:
            se = getattr(gm, "search_entry_point", None)
            if se:
                # Probe a few likely attribute names across versions
                for attr in ("suggested_queries", "suggestedQuestions", "suggestions"):
                    maybe = getattr(se, attr, None)
                    if isinstance(maybe, (list, tuple)):
                        suggestions = [str(x) for x in maybe]
                        break

    # ---- 6) Return structured result ----
    return {
        "answer": answer,
        "sources": sources,            # Markdown links: ["[Title](https://...)"]
        "suggestions": suggestions,    # Optional "Related searches"
        "disclaimer": "This is general information, not medical advice."
    }



# --- quick manual test ---
if __name__ == "__main__":
    res = get_parent_answer("Im so fed up with dyslexia, what should I do?")
    print("\nAnswer:\n", res["answer"])
    # if res["sources"]:
    #     print("\n Sources:")
    #     for s in res["sources"]:
    #         print("-", s)
