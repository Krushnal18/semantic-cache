import numpy as np, torch
from transformers import AutoTokenizer, AutoModel

NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class Encoder:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(NAME)
        self.model = AutoModel.from_pretrained(NAME).eval()
        self.dim = self.model.config.hidden_size

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                b = self.tok(texts[i:i+32], padding=True, truncation=True,
                             max_length=128, return_tensors="pt")
                h = self.model(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)     # mean pooling
                e = torch.nn.functional.normalize(e, dim=1)       # unit length
                out.append(e.numpy())
        return np.vstack(out)