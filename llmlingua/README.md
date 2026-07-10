# LLMLingua Integration

Neural prompt compression using Microsoft's LLMLingua-2 for structured token reduction.

## How It Works

1. **Load the LLMLingua-2 model** (`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`)
2. **Format the input** as a context + instruction + question bundle
3. **Compress to target token budget** while preserving answer-critical information
4. **Fallback to semantic pruning** if LLMLingua is unavailable or fails

## Trade-offs

| Pros | Cons |
|------|------|
| Neural compression preserves meaning | Requires GPU for optimal speed |
| Can achieve 65%+ compression | Model download required (~400MB) |
| Handles complex dependencies | Slower than regex/semantic methods |

## Usage

```python
from llmlingua_prune import llmlingua_prune

context = llmlingua_prune(document_text, question, target_ratio=0.35)
```

## Parameters

- `text` (str): The full document text
- `question` (str): The user's query
- `target_ratio` (float, default=0.35): Target compression ratio (0.35 = keep 35% of tokens)

## Requirements

- `llmlingua` package (`pip install llmlingua`)
- Falls back to semantic chunk pruning if unavailable
