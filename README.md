# Context Optimization

A toolkit for context engineering and prompt compression in LLM applications. Benchmarks three compression methods against a raw baseline, measuring both efficiency (tokens, latency, cost) and quality (LLM-as-a-Judge scoring).

## Project Structure

```
Context Optimization/
├── Adv1_Context_Quarantine.md                    # Context Quarantine guide (LangGraph)
├── Pruning_Ques_Prompt.txt                       # Needle-in-haystack test question
├── context_engineering_pruning_dashboard.ipynb   # Main benchmark notebook
├── nexus_Q2_earnings_projections.pdf             # Sample test document
├── results.txt                                   # Benchmark results
│
├── naive_regex_pruning/          # Simplest: keyword sentence filtering
│   ├── README.md
│   └── naive_regex_pruning.ipynb
│
├── semantic_chunk_pruning/       # Intermediate: TF-IDF chunk ranking
│   ├── README.md
│   └── semantic_chunk_pruning.ipynb
│
└── llmlingua/                    # Advanced: neural token compression
    ├── README.md
    └── llmlingua.ipynb
```

## Method Comparison

| Method | Approach | Tech | Speed | Quality |
|--------|----------|------|-------|---------|
| **Naive Regex** | Keyword sentence filtering | `re` (stdlib) | Fastest | Basic |
| **Semantic Chunk** | TF-IDF + cosine similarity | `scikit-learn` | Medium | Good |
| **LLMLingua** | Neural token compression (XLM-RoBERTa) | `microsoft/llmlingua-2-xlm-roberta-large` | Slowest | Best |

## Getting Started

1. Open `context_engineering_pruning_dashboard.ipynb` in Google Colab or Jupyter
2. Run all setup cells
3. Upload a document and enter a question
4. Click **Run Full Benchmark**

## Requirements

- Python 3.11+
- Groq API key (free inference via api.groq.com)
- Packages: `openai`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `pypdf`, `llmlingua`

## Test Question

> "What is the exact projected infrastructure capital expenditure for Q3 2026, and what two specific factors (one expense and one offset) contribute to this financial change?"

**Gold Standard Answer:**
- Q3 2026 infra CAPEX = $14.73M
- Expense: Frankfurt server farm expansion for EU AI Act data residency
- Offset: $2.1M green energy tax rebate

## License

MIT
