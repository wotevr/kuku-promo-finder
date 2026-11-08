"""
Which Gemini model answers fastest right now? (free-tier models are sometimes overloaded)
Sends one tiny request to each model and prints the time / error. Writes speed_test.txt.

    python speed_test.py gemini-3.5-flash gemini-flash-lite-latest ...
"""

import sys
import time

import requests

import llm

models = sys.argv[1:] or ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-flash-lite-latest", "gemini-3.5-flash-lite"]
s = llm.settings()
lines = []
for m in models:
    body = {"contents": [{"role": "user", "parts": [{"text": 'Reply with exactly this JSON: {"ok": true}'}]}],
            "generationConfig": {"responseMimeType": "application/json"}}
    t = time.time()
    try:
        r = requests.post(f"{s['base_url']}/models/{m}:generateContent", json=body,
                          headers={"x-goog-api-key": s["api_key"]}, timeout=90)
        msg = "OK" if r.ok else f"HTTP {r.status_code}: {r.text[:120]}"
    except requests.RequestException as e:
        msg = f"ERROR {type(e).__name__}"
    line = f"{m:<32} {time.time() - t:6.1f}s  {msg}"
    print(line)
    lines.append(line)
open("speed_test.txt", "w", encoding="utf-8").write("\n".join(lines))
