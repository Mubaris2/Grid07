"""Routing logic to match posts to bot personas using ChromaDB.

This module builds a fresh in-memory persona collection via
`get_persona_collection` and queries it with incoming post text. Distances
returned by ChromaDB (L2) are converted to a cosine-like similarity score
using the formula `1 - (distance / 2)` and filtered by a threshold.
"""

# --- Imports ---------------------------------------------------------------
from typing import List, Dict

from vector_store import get_persona_collection


# --- Core routing function ------------------------------------------------
def route_post_to_bots(post_content: str, threshold: float = 0.5) -> List[Dict]:
    """Route a post to matching bot personas.

    Args:
        post_content: The text of the post to match.
        threshold: Minimum similarity (0-1) to include a bot in results.

    Returns:
        A list of dicts with keys `bot_id` and `similarity`, sorted by
        similarity descending.
    """

    # Rebuild a fresh in-memory collection for a clean query.
    collection = get_persona_collection()

    # Query the collection. We ask for up to 3 results (we have 3 personas).
    result = collection.query(query_texts=[post_content], n_results=3)

    # ChromaDB returns lists per query; we only queried a single text so
    # take the first (and only) row of ids/distances.
    ids_row = result.get("ids", [[]])[0]
    distances_row = result.get("distances", [[]])[0]

    matches: List[Dict] = []

    # Convert distances to similarity and filter by threshold.
    for bot_id, distance in zip(ids_row, distances_row):
        # Convert L2 distance to a cosine-like similarity score.
        # For normalized embeddings there is a simple linear relationship
        # used here: similarity = 1 - (distance / 2).
        # Note: for exact cosine from Euclidean distance you'd normally use
        # `cosine = 1 - (distance**2) / 2` when distance is the L2 norm.
        # The chosen formula follows the project's specified conversion.
        try:
            similarity = 1.0 - (float(distance) / 2.0)
        except Exception:
            # If distance can't be parsed, skip this result.
            continue

        if similarity >= threshold:
            matches.append({"bot_id": bot_id, "similarity": float(similarity)})

    # Sort by similarity descending before returning.
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches


# --- Quick manual tests when run directly -------------------------------
if __name__ == "__main__":
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits all time high, Fed signals rate cuts incoming.",
        "Deforestation is accelerating due to corporate greed.",
    ]

    for post in test_posts:
        matches = route_post_to_bots(post)
        print("Post:", post)
        if not matches:
            print("  No bots matched above threshold")
        else:
            for m in matches:
                print(f"  {m['bot_id']}: {m['similarity']:.4f}")
        print()
