import numpy as np
from embed import Encoder
from data import FAQS, TEST

enc = Encoder()
F = enc.encode([q for q, _ in FAQS])
Q = enc.encode([q for q, _ in TEST])
S = Q @ F.T
top, sim = S.argmax(1), S.max(1)
labels = [l for _, l in TEST]

print("thr   hit_rate  false_hits  total_hits")
for t in np.arange(0.50, 0.96, 0.05):
    pos = false = 0
    for i, l in enumerate(labels):
        if sim[i] >= t:
            if l is not None and top[i] == l:
                pos += 1
            else:
                false += 1
    print(f"{t:.2f}  {pos/60:7.0%}  {false:10d}  {pos+false:10d}")

# sweep table dekhne ke baad T badlo
T = 0.75
print(f"\n--- Problems at threshold {T} ---")
for i, (q, l) in enumerate(TEST):
    hit = sim[i] >= T
    if l is not None and not hit:
        print(f"MISSED     {sim[i]:.2f}  {q}")
    elif hit and (l is None or top[i] != l):
        print(f"FALSE HIT  {sim[i]:.2f}  {q}  ->  {FAQS[top[i]][0]}")