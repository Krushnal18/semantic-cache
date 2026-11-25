# Semantic Cache for an FAQ Bot

Answers repeat support questions (English + Hinglish) from a cache by meaning, not exact text, and calls the LLM only on a miss.

**Result:** at threshold 0.60, the cache served **50% of paraphrased questions (30/60)** with **2 false hits** on a 100-question labelled test set.

## Problem
Support bots get the same question in many wordings ("refund kab milega?" vs "When will I get my refund?"). An exact-match cache misses these, so every one triggers a slow, paid LLM call.

## How it works
1. Embed each question with `paraphrase-multilingual-MiniLM-L12-v2` (mean pooling, L2-normalised vectors).
2. Find the nearest cached question by cosine similarity (a NumPy dot product, since vectors are normalised).
3. If similarity >= threshold, return the cached answer (hit). Otherwise call the LLM (llama3.2 via Ollama) and return its answer (miss).
4. Every request is logged to `requests_log.csv` (question, hit/miss, similarity, latency, tokens).

## Evaluation
- 30 FAQs (e-commerce support).
- 100 labelled test questions: 60 paraphrases (30 English, 30 romanized Hinglish) and 40 unrelated questions that should not hit.
- **False hit** = a hit on an unrelated question, or a hit that returned the wrong FAQ.
- Cache is fixed during evaluation (`learn=False`), so misses do not pollute later results.

| Threshold | Hit rate | False hits |
|---|---|---|
| 0.50 | 63% | 12 |
| 0.55 | 62% | 7 |
| **0.60** | **50%** | **2** |
| 0.65 | 40% | 2 |
| 0.70 | 30% | 1 |
| 0.75 | 25% | 0 |
| 0.80 | 15% | 0 |
| 0.85 | 7% | 0 |

Full sweep output, including every missed question: [`sweep_output.txt`](sweep_output.txt).

**Why 0.60:** it has the highest hit rate with at most 2 false hits. Below it, false hits rise quickly (7 at 0.55, 12 at 0.50). Above it, savings fall. 0.75 is the safest setting (0 false hits) but serves only 25%.

## Failure analysis
- **Romanized Hinglish scores lowest:** "kya COD available hai?" (0.25), "galat product aa gaya" (0.28), "kya aap videsh mein deliver karte ho?" (0.27). A likely reason is that romanized Hindi is rare in the model's training data (hypothesis, not tested).
- **Different vocabulary also misses:** "Where is my package right now?" scored 0.33 against "How do I track my order?".
- **Threshold trade-off:** a lower threshold raises hit rate but also the chance of a confident wrong answer.

## Limitations
- Small, hand-built test set, and the threshold was tuned on the same set, so the numbers are optimistic.
- Cached answers can go stale (production needs a TTL).
- User-specific questions such as "where is my order?" should never be served from a shared cache.
- NumPy exact search is used instead of FAISS: 30 vectors do not need an index, and FAISS was blocked by Windows Application Control on my machine. At scale, use FAISS/HNSW or Redis.

## Project structure
| File | Purpose |
|---|---|
| `data.py` | FAQs and the labelled test set |
| `embed.py` | Embedding model, mean pooling, normalisation |
| `cache.py` | `SemanticCache` class and the Ollama LLM call |
| `sweep.py` | Threshold sweep and failure listing (no LLM calls) |
| `run.py` | Full run with the LLM: hit rate, false hits, latency |
| `app.py` | Gradio demo (not deployed yet) |

## Run
```bash
pip install -r requirements.txt
ollama pull llama3.2
python sweep.py
python run.py 0.60
```

## Next steps
- Add extra phrasings per FAQ to improve Hinglish hits.
- Try a stronger multilingual embedding model.
- Re-check the threshold on a separate held-out test set.
- Deploy the Gradio demo.
