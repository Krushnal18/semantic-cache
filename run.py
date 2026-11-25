import sys, numpy as np
from data import FAQS, TEST
from cache import SemanticCache, call_llm

T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75

print("Warming up Ollama...")
call_llm("warm up")                      # first call loads the model, so keep it out of timings

cache = SemanticCache(threshold=T)
for i, (q, a) in enumerate(FAQS):
    cache.add(q, a, faq_id=i)

pos = false = 0
tokens_used = 0
lat_hit, lat_miss = [], []
for q, label in TEST:
    ans, hit, sim, entry, ms = cache.ask(q, learn=False)   # learn=False keeps results clean
    (lat_hit if hit else lat_miss).append(ms)
    if hit:
        if label is not None and entry["faq_id"] == label:
            pos += 1
        else:
            false += 1

print(f"\nThreshold {T}")
print(f"Hit rate on paraphrases: {pos}/60 = {pos/60:.0%}")
print(f"False hits: {false}")
print(f"Total hits: {len(lat_hit)} | misses (LLM calls): {len(lat_miss)}")
if lat_hit:
    print(f"Median latency  hit: {np.median(lat_hit):.1f} ms")
if lat_miss:
    print(f"Median latency  miss: {np.median(lat_miss):.0f} ms")