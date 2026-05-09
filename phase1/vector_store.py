"""Create an in-memory Chroma collection containing bot persona documents.

This module loads environment variables (for later phases), defines three
bot personas, builds an ephemeral in-memory Chroma collection using a
SentenceTransformer embedding function, and exposes a helper to rebuild
and return the collection on demand.
"""

# --- Load .env variables (for later phases) ---------------------------------
from dotenv import load_dotenv
import os

load_dotenv()
# Load the GROQ API key into memory (not used here, but available later).
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# --- External dependencies --------------------------------------------------
import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


# --- Define bot personas ---------------------------------------------------
# Each persona is a string describing a particular worldview and tone.
PERSONAS = {
    "bot_a": (
        "I believe AI and crypto will solve all human problems. "
        "I am highly optimistic about technology, Elon Musk, and space exploration. "
        "I dismiss regulatory concerns."
    ),
    "bot_b": (
        "I believe late-stage capitalism and tech monopolies are destroying society. "
        "I am highly critical of AI, social media, and billionaires. "
        "I value privacy and nature."
    ),
    "bot_c": (
        "I strictly care about markets, interest rates, trading algorithms, and making money. "
        "I speak in finance jargon and view everything through the lens of ROI."
    ),
}


# --- Helper to build and return a fresh in-memory collection ----------------
def get_persona_collection():
    """Rebuild and return an in-memory Chroma collection with the personas.

    Returns:
        chromadb.api.models.Collection: freshly created in-memory collection
    """

    # Create an in-memory Chroma client (ephemeral by default).
    client = chromadb.Client()

    # Use a SentenceTransformer-based embedding function (small, fast model).
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Create the collection with the embedding function attached. If a
    # collection with the same name already exists (e.g. persistent backend),
    # remove it and recreate to ensure a fresh in-memory collection.
    try:
        collection = client.create_collection(
            name="bot_personas", embedding_function=embedding_fn
        )
    except chromadb.errors.InternalError:
        # Delete existing collection and recreate it to avoid "already exists" errors.
        try:
            client.delete_collection("bot_personas")
        except Exception:
            # If deletion fails for any reason, attempt to get the existing collection.
            collection = client.get_collection("bot_personas")
        else:
            collection = client.create_collection(
                name="bot_personas", embedding_function=embedding_fn
            )

    # Add persona documents to the collection. IDs match persona keys.
    ids = ["bot_a", "bot_b", "bot_c"]
    documents = [PERSONAS[k] for k in ids]
    collection.add(ids=ids, documents=documents)

    return collection


# --- Module verification when executed directly -----------------------------
if __name__ == "__main__":
    # Rebuild the collection and print all stored documents to verify.
    col = get_persona_collection()
    # `collection.get()` returns stored ids/documents/embeddings; print it.
    print(col.get())
