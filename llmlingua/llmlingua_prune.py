import importlib
import re
import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Check if LLMLingua is available
LLMLINGUA_AVAILABLE = False
try:
    importlib.import_module("llmlingua")
    LLMLINGUA_AVAILABLE = True
except ImportError:
    pass

# Lazy singleton for PromptCompressor
PROMPT_COMPRESSOR = None


def token_count(text: str) -> int:
    """Fast token count approximation (words * 1.3)."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def sentence_split(text: str) -> List[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


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


def semantic_chunk_prune(text: str, question: str, top_k: int = 5) -> str:
    """Fallback: TF-IDF based semantic pruning."""
    chunks = chunk_text(text)
    if not chunks:
        return ""

    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vect.fit([question] + chunks)

    q_vec = vect.transform([question])
    c_vec = vect.transform(chunks)
    scores = cosine_similarity(q_vec, c_vec).flatten()

    idx = np.argsort(scores)[::-1][:min(top_k, len(chunks))]
    selected = [chunks[i] for i in idx]
    return "\n\n".join(selected)


def llmlingua_prune(text: str, question: str, target_ratio: float = 0.35) -> str:
    """
    Compress context using LLMLingua-2 neural prompt compression.

    Args:
        text: The full document text
        question: The user's query
        target_ratio: Target compression ratio (0.35 = keep 35% of tokens)

    Returns:
        Compressed text preserving answer-critical information
    """
    global PROMPT_COMPRESSOR

    target_words = max(150, int(len(text.split()) * target_ratio))
    target_tokens = max(200, int(token_count(text) * target_ratio))

    if LLMLINGUA_AVAILABLE:
        try:
            if PROMPT_COMPRESSOR is None:
                from llmlingua import PromptCompressor
                PROMPT_COMPRESSOR = PromptCompressor(
                    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                    use_llmlingua2=True,
                    device_map="cpu"
                )

            compression_input = f"Context:\n{text}\n\nQuestion: {question}"
            compressed = PROMPT_COMPRESSOR.compress_prompt(
                context=compression_input,
                instruction="Keep only information required to answer the question accurately.",
                question=question,
                target_token=target_tokens
            )

            if isinstance(compressed, dict):
                compressed_text = (
                    compressed.get("compressed_prompt")
                    or compressed.get("prompt")
                    or compressed.get("text")
                    or ""
                )
            else:
                compressed_text = str(compressed)

            if compressed_text.strip():
                return " ".join(compressed_text.split()[:target_words])
        except Exception:
            pass

    # Fallback to semantic pruning
    reduced = semantic_chunk_prune(text, question, top_k=5)
    words = reduced.split()
    return " ".join(words[:target_words])


if __name__ == "__main__":
    sample_text = (
        "The company reported strong Q3 earnings. Revenue grew by 15% year over year. "
        "The expansion of the Frankfurt server farm cost $14.73 million. This was partially "
        "offset by a $2.1 million green energy tax rebate. Employee headcount remained stable. "
        "Management raised full-year guidance citing strong demand in European markets."
    )
    sample_question = "What is the projected infrastructure capital expenditure for Q3 2026?"

    result = llmlingua_prune(sample_text, sample_question, target_ratio=0.5)
    print("Pruned output:")
    print(result)
