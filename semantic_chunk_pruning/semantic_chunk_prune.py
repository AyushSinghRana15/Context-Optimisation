import re
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_text(text: str, chunk_size_words: int = 180, overlap_words: int = 40) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size_words - overlap_words)
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size_words]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


def semantic_chunk_prune(text: str, question: str, top_k: int = 8) -> str:
    """
    Prune context by ranking chunks via TF-IDF cosine similarity to the question.

    Args:
        text: The full document text
        question: The user's query
        top_k: Number of top chunks to keep

    Returns:
        Pruned text with most semantically relevant chunks
    """
    chunks = chunk_text(text)
    if not chunks:
        return ""

    # Build TF-IDF vectors for question and all chunks
    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vect.fit([question] + chunks)

    q_vec = vect.transform([question])
    c_vec = vect.transform(chunks)

    # Rank by cosine similarity and keep top-k
    scores = cosine_similarity(q_vec, c_vec).flatten()
    idx = np.argsort(scores)[::-1][:min(top_k, len(chunks))]

    selected = [chunks[i] for i in idx]
    return "\n\n".join(selected)


if __name__ == "__main__":
    sample_text = (
        "The company reported strong Q3 earnings. Revenue grew by 15% year over year. "
        "The expansion of the Frankfurt server farm cost $14.73 million. This was partially "
        "offset by a $2.1 million green energy tax rebate. Employee headcount remained stable. "
        "Management raised full-year guidance citing strong demand in European markets."
    )
    sample_question = "What is the projected infrastructure capital expenditure for Q3 2026?"

    result = semantic_chunk_prune(sample_text, sample_question, top_k=3)
    print("Pruned output:")
    print(result)
