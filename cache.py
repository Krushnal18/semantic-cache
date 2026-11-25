import time, csv, os, numpy as np, ollama
from embed import Encoder

LLM_MODEL = "llama3.2"          # slow laptop ho to "llama3.2:1b"
LOG = "requests_log.csv"

def call_llm(question):
    r = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a customer support bot for an Indian online store. Answer briefly in 2 sentences."},
            {"role": "user", "content": question},
        ],
        options={"num_predict": 150},
    )
    tokens = r["prompt_eval_count"] + r["eval_count"]
    return r["message"]["content"], tokens

class SemanticCache:
    def __init__(self, threshold=0.75):
        self.enc = Encoder()
        self.vecs = np.zeros((0, self.enc.dim), dtype="float32")
        self.entries = []
        self.threshold = threshold
        if not os.path.exists(LOG):
            with open(LOG, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["question", "hit", "similarity", "matched_q", "latency_ms", "tokens"])

    def embed(self, texts):
        return self.enc.encode(texts).astype("float32")

    def add(self, q, a, faq_id=None):
        self.vecs = np.vstack([self.vecs, self.embed([q])])
        self.entries.append({"q": q, "a": a, "faq_id": faq_id})

    def lookup(self, q):
        if not self.entries:
            return None, -1.0
        sims = self.vecs @ self.embed([q])[0]      # dot product == cosine (normalised)
        i = int(sims.argmax())
        return self.entries[i], float(sims[i])

    def ask(self, q, learn=True):
        t0 = time.perf_counter()
        entry, sim = self.lookup(q)
        if entry and sim >= self.threshold:
            ans, hit, tokens, matched = entry["a"], True, 0, entry["q"]
        else:
            ans, tokens = call_llm(q)
            hit, matched = False, ""
            if learn:
                self.add(q, ans)
        ms = (time.perf_counter() - t0) * 1000
        with open(LOG, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([q, hit, round(sim, 4), matched, round(ms, 1), tokens])
        return ans, hit, sim, entry, ms