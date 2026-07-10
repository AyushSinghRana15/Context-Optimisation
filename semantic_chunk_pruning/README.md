# Semantic Chunk Pruning

TF-IDF based semantic similarity ranking for context compression.

## How It Works

1. **Chunk the document** into overlapping windows (default: 180 words, 40 word overlap)
2. **Vectorize** both the question and all chunks using TF-IDF with bigrams
3. **Compute cosine similarity** between the question vector and each chunk vector
4. **Rank chunks** by similarity score and keep the top-k most relevant

## Trade-offs

| Pros | Cons |
|------|------|
| Captures semantic relevance better than regex | Requires scikit-learn dependency |
| Handles synonyms/partial matches via n-grams | TF-IDF is less nuanced than neural embeddings |
| Overlapping chunks preserve local context | Chunk size is a hyperparameter to tune |

## Usage

```python
from semantic_chunk_prune import semantic_chunk_prune

context = semantic_chunk_prune(document_text, question, top_k=8)
```

## Parameters

- `text` (str): The full document text
- `question` (str): The user's query
- `top_k` (int, default=8): Number of top chunks to keep
