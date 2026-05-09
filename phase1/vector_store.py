"""
phase1.vector_store

Responsible for loading OpenAI credentials, creating an in-memory
ChromaDB collection for bot personas using OpenAI embeddings, and
exposing the collection via `get_persona_collection()`.
"""

# --- Load environment variables (OpenAI key) -----------------------------
from dotenv import load_dotenv
import os

load_dotenv()  # read .env (or .env.example) into environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --- Define bot personas -------------------------------------------------
# A simple dictionary of three distinct persona strings. These are the
# documents we'll insert into the vector store so other phases can retrieve
# or embed them.
PERSONAS = {
    "bot_a": (
        "Tech Maximalist persona: relentlessly optimistic about technology's "
        "ability to solve problems. Speaks in forward-looking, visionary terms, "
        "emphasizes cutting-edge AI, decentralization, and transformative "
        "infrastructure."
    ),
    "bot_b": (
        "Doomer/Skeptic persona: cautiously pessimistic, questions techno-optimism, "
        "focuses on systemic risks, unintended consequences, and historical "
        "failures. Uses skeptical, reality-check language."
    ),
    "bot_c": (
        "Finance Bro persona: jargon-heavy, ROI-focused, talks about market "
        "opportunities, risk-adjusted returns, and scaling businesses. Conversational "
        "tone with finance metaphors and performance metrics."
    ),
}


# --- ChromaDB in-memory client and OpenAI embedding function -------------
# Use an in-memory Chroma client and the OpenAI embedding function so that
# downstream phases can import the loaded collection.
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Create an in-memory Chroma client (default behavior)
client = chromadb.Client()

# Configure the OpenAI embedding function. The exact model name may be
# adjusted; `text-embedding-3-small` is a reasonable default at time of writing.
embedding_function = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)


# --- Create or get the collection named "bot_personas" --------------------
# We attempt to create the collection with the embedding function. If the
# collection already exists (e.g., module re-imports during development), we
# fall back to getting the existing collection.
COLLECTION_NAME = "bot_personas"
try:
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
except Exception:
    # Fall back to retrieving existing collection if creation fails
    collection = client.get_collection(name=COLLECTION_NAME)


# --- Insert persona documents into the collection ------------------------
# Add the three persona documents with explicit ids so they can be referenced
# by other phases. To avoid duplicates on repeated imports, check for
# existing ids before adding.
ids = list(PERSONAS.keys())
documents = list(PERSONAS.values())

def _ensure_personas_inserted():
    try:
        existing = collection.get(ids=ids).get("ids", [])
    except Exception:
        existing = []

    to_add_ids = []
    to_add_docs = []
    for i, _id in enumerate(ids):
        if _id not in existing:
            to_add_ids.append(_id)
            to_add_docs.append(documents[i])

    if to_add_ids:
        collection.add(ids=to_add_ids, documents=to_add_docs)


# Ensure insertion on import
_ensure_personas_inserted()


# --- Public accessor -----------------------------------------------------
def get_persona_collection():
    """Return the in-memory Chroma collection containing the personas.

    Other project phases should import and call this function to access the
    preloaded collection.
    """

    return collection


# --- Quick verification when run as a script -----------------------------
if __name__ == "__main__":
    # Print all stored documents to verify successful insertion
    try:
        stored = collection.get(ids=ids)
        print("Stored persona documents:")
        for _id, doc in zip(stored.get("ids", []), stored.get("documents", [])):
            print(f"- {_id}: {doc}")
    except Exception as e:
        print("Error retrieving stored personas:", e)
