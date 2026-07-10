# Semantic Chunk Pruning

A TF-IDF based approach that scores document chunks by semantic similarity to the user's question. Uses term frequency statistics to find relevant passages even when wording differs.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Vectorization | `scikit-learn` (`TfidfVectorizer`) |
| Similarity | `cosine_similarity` from scikit-learn |
| Text processing | `re` (regex) for chunking |
| Runtime | Jupyter / Colab |

## Approach

1. **Chunk** the document into overlapping sliding windows (default: 180 words, 40 overlap)
2. **Vectorize** both the question and all chunks using TF-IDF with unigrams + bigrams
3. **Score** each chunk by cosine similarity to the question vector
4. **Rank** — sort chunks by similarity score (descending)
5. **Select** top-k chunks (default 8)
6. **Return** selected chunks joined by double newlines

The overlap preserves context across chunk boundaries so no passage is split awkwardly.

## Functionalities

- **Overlapping chunks**: 40-word overlap prevents cutting off mid-paragraph
- **Bigram support**: Captures phrases like "infrastructure CAPEX" as single features
- **Semantic ranking**: TF-IDF captures partial matches and related terms that pure keyword matching misses
- **Top-k selection**: Configurable number of chunks to keep
- **Idiom-aware**: If the question says "server farm" and the doc says "data center", bigrams still catch related content

## Limitations

- TF-IDF is bag-of-words — word order within a chunk is lost
- Chunk size is a fixed hyperparameter
- Slower than regex (needs vectorization of all chunks)
