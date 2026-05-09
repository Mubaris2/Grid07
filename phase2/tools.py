"""Utility tools for phase2: mock search tool and LLM initializer.

This module provides a `mock_searxng_search` tool that returns hardcoded
headline strings based on simple keyword matching, and initializes a
`ChatGroq` LLM instance for later use.
"""

# --- Load environment -----------------------------------------------------
from dotenv import load_dotenv

# Load environment variables from `.env` at module import time.
load_dotenv()


# --- External imports ----------------------------------------------------
from langchain_groq import ChatGroq
from langchain_core.tools import tool


# --- Mock search tool ----------------------------------------------------
@tool
def mock_searxng_search(query: str) -> str:
    """Searches the web for recent news headlines based on a query.

    The function lowercases the query and uses simple keyword matching to
    return a hardcoded headline string representative of recent news.
    """
    q = (query or "").lower()

    # Check keywords in prioritized order and return matching headlines.
    if "crypto" in q or "bitcoin" in q:
        return (
            "Bitcoin hits new all-time high amid regulatory ETF approvals. "
            "Ethereum staking yields hit 8%. SEC greenlights spot crypto ETFs."
        )
    elif "ai" in q or "openai" in q or "llm" in q:
        return (
            "OpenAI releases GPT-5 with autonomous agent capabilities. "
            "EU AI Act enforcement begins. 40% of junior dev roles at risk per McKinsey report."
        )
    elif "market" in q or "stock" in q or "fed" in q:
        return (
            "Fed signals two rate cuts in 2025. S&P 500 hits record high. "
            "Goldman Sachs upgrades tech sector to overweight."
        )
    elif "climate" in q or "environment" in q:
        return (
            "UN warns of irreversible tipping points by 2027. Carbon credit markets surge. "
            "Oil majors post record profits despite green pledges."
        )
    elif "elon" in q or "tesla" in q or "space" in q:
        return (
            "SpaceX Starship completes Mars trajectory test. Tesla FSD v13 passes regulatory review. "
            "Musk acquires 10% stake in major AI chipmaker."
        )
    elif "privacy" in q or "surveillance" in q or "big tech" in q:
        return (
            "Meta fined $2B for GDPR violations. Apple expands on-device AI to protect user data. "
            "EU passes landmark digital rights bill."
        )
    elif "regulation" in q or "government" in q or "policy" in q:
        return (
            "US Senate passes sweeping AI regulation bill. FTC targets big tech mergers. "
            "Global AI governance summit scheduled for Geneva."
        )

    # Default fallback headline when no keywords matched.
    return (
        "Global markets show mixed signals. Tech innovation continues at rapid pace. "
        "Geopolitical tensions affect supply chains."
    )


# --- Plain-callable wrapper for local testing --------------------------------
def mock_searxng_search_func(query: str) -> str:
    """Plain callable wrapper for the tool object returned by `@tool`.

    The `@tool` decorator often returns a StructuredTool which is not
    directly callable. This wrapper calls the appropriate attribute
    (`run` or the original function) so the tool can be used in simple
    scripts and tests.
    """
    # Prefer the structured tool's `run` method if available.
    if hasattr(mock_searxng_search, "run"):
        return mock_searxng_search.run(query)
    # Fallback: if the tool is directly callable, call it.
    if callable(mock_searxng_search):
        return mock_searxng_search(query)
    # Fallback: if the decorator preserved the original function as `func`.
    if hasattr(mock_searxng_search, "func"):
        return mock_searxng_search.func(query)
    raise TypeError("mock_searxng_search is not callable or runnable")


# --- LLM initialization --------------------------------------------------
# Initialize a module-level ChatGroq LLM instance for downstream use.
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)


# --- Quick manual tests when run directly -------------------------------
if __name__ == "__main__":
    test_queries = ["crypto market", "openai llm", "climate policy"]
    for q in test_queries:
        print("Query:", q)
        # Use the plain-callable wrapper for testing.
        print("Result:", mock_searxng_search_func(q))
        print()
