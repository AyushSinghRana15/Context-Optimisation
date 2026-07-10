# Context Optimization

A comprehensive toolkit for context engineering and prompt compression techniques for LLMs.

## Overview

This project explores various strategies for optimizing the context window sent to Large Language Models. As context windows grow larger, simply stuffing them with all available data leads to performance degradation (context distraction, context clash). This toolkit provides benchmarking tools and implementations for three prompt compression methods.

## Project Structure

```
Context Optimization/
├── Adv1_Context_Quarantine.md          # Guide on Context Quarantine with LangGraph
├── Pruning_Ques_Prompt.txt            # Needle-in-haystack test question
├── context_engineering_pruning_dashboard.ipynb  # Main benchmark notebook
├── nexus_Q2_earnings_projections.pdf  # Sample document for testing
├── results.txt                        # Benchmark results
├── naive_regex_pruning/
│   ├── README.md
│   └── naive_regex_pruning.ipynb      # Keyword-based sentence filtering
├── semantic_chunk_pruning/
│   ├── README.md
│   └── semantic_chunk_pruning.ipynb   # TF-IDF cosine similarity ranking
└── llmlingua/
    ├── README.md
    └── llmlingua.ipynb                # Neural prompt compression
```

## Compression Methods

| Method | Approach | Speed | Quality |
|--------|----------|-------|---------|
| **Naive Regex** | Keyword-based sentence filtering | Fast | Basic |
| **Semantic Chunk** | TF-IDF cosine similarity ranking | Medium | Good |
| **LLMLingua** | Neural prompt compression (LLMLingua-2) | Slow | Best |

## Getting Started

1. Open `context_engineering_pruning_dashboard.ipynb` in Google Colab or Jupyter
2. Run all setup cells
3. Upload a document and enter your question
4. Click **Run Full Benchmark**

## Requirements

- Python 3.11+
- Google API key (for Gemini LLM)
- Packages: `openai`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `pypdf`, `llmlingua`

## Test Question

The included test question targets a specific data point buried in the PDF:

> "What is the exact projected infrastructure capital expenditure for Q3 2026, and what two specific factors (one expense and one offset) contribute to this financial change?"

**Gold Standard Answer:**
- Q3 2026 infra CAPEX = $14.73M
- Expense: Frankfurt server farm expansion for EU AI Act data residency
- Offset: $2.1M green energy tax rebate

## License

MIT
