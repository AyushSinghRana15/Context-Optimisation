import importlib, os, re, time, json, textwrap, time as time_module
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from pypdf import PdfReader
from io import BytesIO

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PDF_PATH = "nexus_Q2_earnings_projections.pdf"
QUESTION = (
    "What is the exact projected infrastructure capital expenditure for Q3 2026, "
    "and what two specific factors (one expense and one offset) contribute to this financial change?"
)

API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
if not API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set")

client = OpenAI(api_key=API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

LLMLINGUA_AVAILABLE = False
try:
    importlib.import_module("llmlingua")
    LLMLINGUA_AVAILABLE = True
except Exception:
    pass

PROMPT_COMPRESSOR = None

# --- PDF loading ---
print("Loading PDF...")
with open(PDF_PATH, "rb") as f:
    reader = PdfReader(f)
    raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
raw_text = re.sub(r"\s+", " ", raw_text).strip()
print(f"Document length: {len(raw_text)} chars | ~{len(raw_text.split())} words")

# --- Helpers ---
def token_count(text: str) -> int:
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

def sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def chunk_text(text: str, chunk_size_words: int = 180, overlap_words: int = 40) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks, step = [], max(1, chunk_size_words - overlap_words)
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size_words]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks

def call_llm(context: str, question: str, model: str = "gemini-2.5-flash-lite" if "GEMINI_MODEL" not in os.environ else os.environ["GEMINI_MODEL"]) -> Tuple[str, int, int, float]:
    prompt = (
        f"You are a precise analyst. Answer strictly from the provided context.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        f"If information is missing, say so clearly."
    )
    for attempt in range(3):
        try:
            start = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You answer with concise, factual, context-grounded responses."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            latency = time.perf_counter() - start
            answer = resp.choices[0].message.content if resp.choices else ""
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", token_count(prompt))
            out_tok = getattr(usage, "completion_tokens", token_count(answer))
            return answer, in_tok, out_tok, latency
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"    API error (attempt {attempt+1}/3): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

# --- Pruning methods ---
def naive_regex_prune(text: str, question: str, max_sentences: int = 30) -> str:
    sentences = sentence_split(text)
    keywords = [w.lower() for w in re.findall(r"[A-Za-z0-9]{2,}", question)]
    stop_words = {"the", "for", "and", "this", "what", "with", "that", "from",
                  "are", "was", "were", "have", "has", "had", "not", "but",
                  "its", "all", "can", "two", "one", "each", "which", "their",
                  "than", "they", "been", "some", "would", "could", "into",
                  "about", "also", "how", "them", "these", "those", "will",
                  "your", "our", "has", "had", "does", "been"}
    keywords = [k for k in keywords if k not in stop_words]
    if not keywords:
        return " ".join(sentences[:max_sentences])
    scored = [(sum(1 for k in keywords if k in s.lower()), s) for s in sentences]
    scored.sort(key=lambda x: -x[0])
    keep = [s for _, s in scored if _ > 0]
    if not keep:
        keep = sentences[:max_sentences]
    return " ".join(keep[:max_sentences])

def semantic_chunk_prune(text: str, question: str, top_k: int = 8) -> str:
    chunks = chunk_text(text)
    if not chunks:
        return ""
    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vect.fit([question] + chunks)
    q_vec = vect.transform([question])
    c_vec = vect.transform(chunks)
    scores = cosine_similarity(q_vec, c_vec).flatten()
    idx = np.argsort(scores)[::-1][:min(top_k, len(chunks))]
    return "\n\n".join(chunks[i] for i in idx)

def llmlingua_prune(text: str, question: str, target_ratio: float = 0.35) -> str:
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
                    device_map="cpu",
                )
            compression_input = f"Context:\n{text}\n\nQuestion: {question}"
            compressed = PROMPT_COMPRESSOR.compress_prompt(
                context=compression_input,
                instruction="Keep only information required to answer the question accurately.",
                question=question,
                target_token=target_tokens,
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
    reduced = semantic_chunk_prune(text, question, top_k=5)
    return " ".join(reduced.split()[:target_words])

# --- Run benchmark ---
methods = {
    "Raw Baseline": lambda t, q: t,
    "Naive Regex": naive_regex_prune,
    "Semantic Chunk": semantic_chunk_prune,
    "LLMLingua": llmlingua_prune,
}

print("\nBuilding method contexts...")
contexts = {}
for name, func in methods.items():
    print(f"  Pruning: {name}")
    contexts[name] = func(raw_text, QUESTION)

original_tokens = token_count(raw_text)
results = []
answers = {}

print("\nRunning LLM answers...")
for name, ctx in contexts.items():
    print(f"  [{name}] sending to Gemini...")
    answer, in_tok, out_tok, latency = call_llm(ctx, QUESTION)
    comp_tokens = token_count(ctx)
    reduction = max(0.0, (1 - (comp_tokens / max(1, original_tokens))) * 100.0)
    if name == "Raw Baseline":
        reduction = 0.0
    answers[name] = answer
    results.append({
        "Method": name,
        "Context Size (tokens)": comp_tokens,
        "Compression %": round(reduction, 2),
        "Input Tokens": in_tok,
        "Output Tokens": out_tok,
        "Latency (s)": round(latency, 3),
        "Answer Preview": textwrap.shorten(answer, width=300, placeholder="..."),
    })
    print(f"    Latency: {latency:.2f}s | Compression: {reduction:.1f}%")

# --- Print results ---
print("\n" + "=" * 100)
print("BENCHMARK RESULTS")
print("=" * 100)
for r in results:
    print(f"\n{'─' * 80}")
    print(f"Method: {r['Method']}")
    print(f"  Context: {r['Context Size (tokens)']} tokens ({r['Compression %']}% compressed)")
    print(f"  Billed: {r['Input Tokens']} in / {r['Output Tokens']} out | Latency: {r['Latency (s)']}s")
    print(f"  Answer: {r['Answer Preview']}")

print("\n" + "=" * 100)
print("GOLD STANDARD (Expected):")
print("  Q3 2026 infra CAPEX = $14.73M")
print("  Expense: Frankfurt server farm expansion for EU AI Act data residency")
print("  Offset: $2.1M green energy tax rebate")
print("=" * 100)
