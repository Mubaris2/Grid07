"""Build a LangGraph state machine for agent persona post generation.

This graph has three nodes:
 - decide_search: decide a topic and search query
 - web_search: call a mock search tool to get recent headlines
 - draft_post: produce a short opinionated post under 280 chars

Each node returns a dict of fields that are merged into the agent state.
"""

# --- Load environment -----------------------------------------------------
from dotenv import load_dotenv

# Load .env at import time
load_dotenv()


# --- External imports ----------------------------------------------------
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import Optional
from phase2.tools import mock_searxng_search

# Helper to produce a message object compatible with the LLM's `invoke` API.
# Prefer the real `SystemMessage` from langchain when available; otherwise
# fall back to a simple dict with `role` and `content`, which langchain_core
# utilities can also coerce into messages.
try:
    from langchain.schema import SystemMessage

    def _make_system_message(content: str):
        return SystemMessage(content=content)
except Exception:
    def _make_system_message(content: str):
        return {"role": "system", "content": content}

import json
import re


# --- Pydantic state model -----------------------------------------------
class AgentState(BaseModel):
    bot_id: str
    persona: str
    search_query: Optional[str] = None
    search_results: Optional[str] = None
    post_content: Optional[str] = None
    topic: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


# --- LLM initialization -------------------------------------------------
# Module-level LLM instance for nodes to call.
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)


# --- Helpers -------------------------------------------------------------
def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output for clean JSON parsing."""
    if not isinstance(text, str):
        text = str(text)
    # Remove triple backtick blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Strip surrounding whitespace
    return text.strip()


def _parse_json_from_response(resp) -> dict:
    """Convert various LLM response objects to JSON dict.

    This attempts to be robust against different response shapes by
    coercing to string, stripping markdown fences, and loading JSON.
    """
    # Coerce to string if needed
    if hasattr(resp, "content"):
        text = resp.content
    elif isinstance(resp, (list, tuple)) and len(resp) > 0:
        # Take first element and attempt to extract text/content
        first = resp[0]
        text = getattr(first, "content", str(first))
    else:
        text = str(resp)

    text = _strip_markdown_fences(text)

    # Try to find a JSON object in the text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract JSON substring with regex
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    # If parsing fails, raise for visibility
    raise ValueError(f"Unable to parse JSON from LLM response: {text}")


# --- Node 1: decide_search ----------------------------------------------
def decide_search(state: AgentState) -> dict:
    """Decide a topic and search query based on the persona.

    Returns a dict with keys `topic` and `search_query`.
    """
    # System prompt instructing the LLM to return ONLY a JSON object.
    system_prompt = (
        f"You are {state.persona}. Decide what topic you want to post about today. "
        "Respond with ONLY a JSON object: {\"topic\": \"...\", \"search_query\": \"...\"}. No extra text."
    )

    # Invoke the LLM with a system-style message compatible with the LLM API
    resp = llm.invoke([_make_system_message(system_prompt)])

    # Parse response into JSON and return required fields
    parsed = _parse_json_from_response(resp)
    return {"topic": parsed.get("topic"), "search_query": parsed.get("search_query")}


# --- Node 2: web_search --------------------------------------------------
def web_search(state: AgentState) -> dict:
    """Perform a web search using the mock search tool.

    Calls the tool's `invoke` method with the query and returns search results.
    """
    # Use the mock search tool's invoke API
    result = mock_searxng_search.invoke({"query": state.search_query})
    # The tool returns a string of headlines
    return {"search_results": result}


# --- Node 3: draft_post -------------------------------------------------
def draft_post(state: AgentState) -> dict:
    """Draft a short opinionated social post based on topic and search results.

    Returns a dict with `post_content`.
    """
    system_prompt = (
        f"You are {state.persona}.\n"
        f"Today's topic: {state.topic}\n"
        f"Recent news context: {state.search_results}\n\n"
        "Write a highly opinionated social media post under 280 characters based on the context above.\n"
        "You MUST respond with ONLY this JSON object and nothing else:\n"
        f"{{\"bot_id\": \"{state.bot_id}\", \"topic\": \"{state.topic}\", \"post_content\": \"your post here\"}}\n"
        "No markdown, no explanation, no extra text. Raw JSON only."
    )

    resp = llm.invoke([_make_system_message(system_prompt)])
    parsed = _parse_json_from_response(resp)
    return {"post_content": parsed.get("post_content")}


# --- Build graph --------------------------------------------------------
def build_graph():
    """Construct and compile the StateGraph with the three nodes.

    Entry: decide_search -> web_search -> draft_post -> END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("decide_search", decide_search)
    graph.add_node("web_search", web_search)
    graph.add_node("draft_post", draft_post)

    # Set entry point
    graph.set_entry_point("decide_search")

    # Add edges: decide_search -> web_search -> draft_post -> END
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search", "draft_post")
    graph.add_edge("draft_post", END)

    # Compile the graph into an executable app
    return graph.compile()


# --- Run for all bots when executed directly ----------------------------
if __name__ == "__main__":
    bots = [
        {
            "bot_id": "bot_a",
            "persona": (
                "I believe AI and crypto will solve all human problems. "
                "I am highly optimistic about technology, Elon Musk, and space exploration. "
                "I dismiss regulatory concerns."
            ),
        },
        {
            "bot_id": "bot_b",
            "persona": (
                "I believe late-stage capitalism and tech monopolies are destroying society. "
                "I am highly critical of AI, social media, and billionaires. "
                "I value privacy and nature."
            ),
        },
        {
            "bot_id": "bot_c",
            "persona": (
                "I strictly care about markets, interest rates, trading algorithms, and making money. "
                "I speak in finance jargon and view everything through the lens of ROI."
            ),
        },
    ]

    app = build_graph()

    # Print phase header
    print("=== PHASE 2: Autonomous Content Engine ===\n")

    for bot in bots:
        # Invoke the compiled graph with an initial AgentState
        result = app.invoke(AgentState(**bot))

        # Prepare output dict and pretty-print using json.dumps
        output = {
            "bot_id": bot["bot_id"],
            "topic": result.get("topic"),
            "post_content": result.get("post_content"),
        }

        print(f"--- Bot: {bot['bot_id']} ---")
        print(json.dumps(output, indent=2))

    # Completion footer
    print("\n=== Phase 2 Complete ===")
