# Naive Regex Pruning

A simple keyword-based sentence filtering approach for context compression.

## How It Works

1. **Tokenize** the document into sentences
2. **Extract keywords** from the user's question (excluding common stop words)
3. **Score each sentence** by counting how many question keywords it contains
4. **Keep top sentences** that match at least one keyword (up to a configurable limit)

## Trade-offs

| Pros | Cons |
|------|------|
| Fast - no ML dependencies | May miss semantically relevant but lexically different sentences |
| Simple to understand and debug | Stop word list is language-specific |
| Deterministic output | Keyword matching is surface-level |

## Usage

```python
from naive_regex_prune import naive_regex_prune

context = naive_regex_prune(document_text, question, max_sentences=30)
```

## Parameters

- `text` (str): The full document text
- `question` (str): The user's query
- `max_sentences` (int, default=30): Maximum sentences to keep
