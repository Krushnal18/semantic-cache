# Semantic Cache for an FAQ Bot

Answers repeat support questions (English + Hinglish) from a cache by meaning, not exact text.

**Result:** at threshold 0.60: 50% hit rate (30/60 paraphrases), 2 false hits, __ ms on a cache hit vs __ ms on an LLM call (llama3.2 via Ollama, CPU laptop).

## How it works
1. Embed each question with `paraphrase-multilingual-MiniLM-L12-v2` (mean pooling, normalised vectors).
2. Find the nearest cached question by cosine similarity (NumPy dot product).
3. If similarity >= threshold, return the cached answer (hit). Otherwise call the LLM (miss).

## Evaluation
30 FAQs, 100 labelled test questions: 60 paraphrases (30 English, 30 Hinglish) and 40 unrelated questions that should not hit.
A false hit = a hit on an unrelated question, or a hit that returned the wrong FAQ.

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

Full output: `sweep_output.txt`.

## Failure analysis
- Romanized Hinglish scores lowest: "kya COD available hai?" (0.25), "galat product aa gaya" (0.28), "kya aap videsh mein deliver karte ho?" (0.27). The model is stronger on English and native-script text.
- Different vocabulary also misses: "Where is my package right now?" scored 0.33 against "How do I track my order?".
- False hits at 0.60: <paste the 2 FALSE HIT lines from sweep_output.txt and explain why>.
- Trade-off: lowering the threshold raises hit rate but also wrong answers (12 false hits at 0.50).

## Run
pip install -r requirements.txt
ollama pull llama3.2
python sweep.py
python run.py 0.60

## Notes
- NumPy exact search is used instead of FAISS: 30 vectors don't need an index, and FAISS was blocked by Windows Application Control on my machine. At scale, use FAISS/HNSW or Redis.
- Cached answers can go stale (needs a TTL), and user-specific questions like "where is my order?" should never be served from a shared cache.
- Next steps: add extra phrasings per FAQ, try a stronger multilingual embedding model.