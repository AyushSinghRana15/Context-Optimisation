import re
from typing import List


def sentence_split(text: str) -> List[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def naive_regex_prune(text: str, question: str, max_sentences: int = 30) -> str:
    """
    Prune context by keeping sentences that match question keywords.

    Args:
        text: The full document text
        question: The user's query
        max_sentences: Maximum sentences to keep

    Returns:
        Pruned text containing only relevant sentences
    """
    sentences = sentence_split(text)

    # Extract keywords from question (words with 3+ chars)
    keywords = [w.lower() for w in re.findall(r'[A-Za-z0-9]{3,}', question)]

    # Common English stop words to exclude
    stop_words = {
        "the", "for", "and", "this", "what", "with", "that", "from",
        "are", "was", "were", "have", "has", "had", "not", "but",
        "its", "all", "can", "two", "one", "each", "which", "their",
        "than", "they", "been", "some", "would", "could", "into",
        "about", "also", "how", "them", "these", "those", "will",
        "your", "our", "does"
    }
    keywords = [k for k in keywords if k not in stop_words]

    if not keywords:
        return " ".join(sentences[:max_sentences])

    # Score sentences by keyword count
    scored = [(sum(1 for k in keywords if k in s.lower()), s) for s in sentences]
    scored.sort(key=lambda x: -x[0])

    # Keep sentences with at least one keyword match
    keep = [s for _, s in scored if _ > 0]

    if not keep:
        keep = sentences[:max_sentences]

    return " ".join(keep[:max_sentences])


if __name__ == "__main__":
    sample_text = (
        "The company reported strong Q3 earnings. Revenue grew by 15% year over year. "
        "The expansion of the Frankfurt server farm cost $14.73 million. This was partially "
        "offset by a $2.1 million green energy tax rebate. Employee headcount remained stable."
    )
    sample_question = "What is the projected infrastructure capital expenditure for Q3 2026?"

    result = naive_regex_prune(sample_text, sample_question)
    print("Pruned output:")
    print(result)
