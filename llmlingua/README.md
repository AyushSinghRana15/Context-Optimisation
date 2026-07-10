# LLMLingua Integration

Neural prompt compression using Microsoft's LLMLingua-2. A trained transformer model removes redundant tokens while preserving question-critical information — the most sophisticated compression method in the toolkit.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Compression model | `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` |
| Library | `llmlingua` (PromptCompressor) |
| Hardware | CPU (fallback) / GPU recommended |
| Runtime | Jupyter / Colab |

## Approach

1. **Compute budget** — target tokens = `max(200, total_tokens * target_ratio)`
2. **Format input** as a structured bundle: instruction + context + question
3. **Compress** via `PromptCompressor.compress_prompt()` — the model scores each token by importance and drops low-value ones
4. **Extract** compressed text from the response
5. **Hard-cap** output to `target_words` for fair comparison
6. **Fallback** — if LLMLingua is unavailable, falls back to semantic chunk pruning

## Functionalities

- **Token-level compression**: Removes redundant words, not just sentences or chunks
- **Question-aware**: Compression is guided by the specific question being asked
- **Configurable ratio**: `target_ratio` controls how aggressive the compression is (0.35 = keep 35%)
- **Graceful fallback**: Degrades to semantic chunk pruning if the model isn't installed
- **Latent understanding**: The model understands which tokens are critical for QA, not just frequency-based importance

## Limitations

- Requires downloading the model (~400MB) on first run
- Slowest method — each call is a forward pass through BERT
- CPU inference can be 2-5x slower than the other methods
