import os, time, json, urllib.request
import numpy as np, gradio as gr
from embed import Encoder
from data import FAQS

enc = Encoder()
F = enc.encode([q for q, _ in FAQS])

def call_llm(question):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return "(No API key set. A real deployment would call the LLM here.)"
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.5-flash:generateContent?key={key}")
    body = {
        "systemInstruction": {"parts": [{"text": "You are a customer support bot for an Indian online store. Answer briefly in 2 sentences."}]},
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "generationConfig": {"maxOutputTokens": 150, "thinkingConfig": {"thinkingBudget": 0}},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["candidates"][0]["content"]["parts"][0]["text"]

def ask(question, threshold):
    t0 = time.perf_counter()
    sims = F @ enc.encode([question])[0]
    i = int(sims.argmax()); sim = float(sims[i])
    if sim >= threshold:
        ans, tag = FAQS[i][1], "HIT (served from cache, no LLM call)"
    else:
        ans, tag = call_llm(question), "MISS (LLM called)"
    ms = (time.perf_counter() - t0) * 1000
    return f"**{tag}**\n\nSimilarity: {sim:.2f} | Nearest FAQ: *{FAQS[i][0]}* | Time: {ms:.0f} ms\n\n{ans}"

demo = gr.Interface(
    fn=ask,
    inputs=[gr.Textbox(label="Ask a question (English or Hinglish)"),
            gr.Slider(0.4, 0.95, value=0.60, step=0.05, label="Similarity threshold")],
    outputs=gr.Markdown(),
    title="Semantic Cache for an FAQ Bot",
    examples=[["refund kab milega?", 0.60], ["Where is my package right now?", 0.60], ["Do you sell iPhones?", 0.60]],
)
demo.launch()