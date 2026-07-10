# Naive Regex Pruning

A keyword-based sentence filtering approach — the simplest form of context compression. It selects only the sentences containing words that match the user's question.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Parsing | `re` (regex) — standard library |
| Dependencies | None (zero external ML deps) |
| Runtime | Jupyter / Colab |

## Approach

1. **Split** the document into sentences using regex (`(?<=[.!?])\s+`)
2. **Extract keywords** from the user's question — keep alphanumeric words of 3+ characters, filter out common stop words
3. **Score** each sentence by counting how many question keywords appear (case-insensitive)
4. **Filter** — keep only sentences with at least one keyword match
5. **Cap** results to `max_sentences` (default 30)
6. **Fallback** — if no keywords match, return the first N sentences

## Functionalities

- **Keyword extraction**: Automatically extracts meaningful terms from any user question
- **Stop word filtering**: Removes ~35 common English words (the, is, at, which, what, etc.)
- **Sentence scoring**: Ranks sentences by keyword match count (descending)
- **Max cap**: Limits output size to prevent oversized context (configurable)
- **Fallback mode**: Returns document start if no keywords match anywhere

## Limitations

- Purely lexical — "car" and "vehicle" are different keywords
- Stop word list is English-only and hardcoded
- Ordering is by relevance score, not original document position
