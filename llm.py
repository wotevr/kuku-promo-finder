"""
The ONLY file that talks to an AI model.

It uses the "OpenAI-compatible" chat API, which Gemini, Groq, OpenRouter, GitHub Models,
OpenAI, Ollama and many others all support. So swapping the model = editing 3 lines in .env:
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
No code change needed.

Every answer is saved to outputs/raw_llm/<name>.json. If that file already exists, the saved
answer is reused instead of calling the API again (unless you run with --fresh). This means a
run that stops halfway (e.g. the free daily quota runs out) continues where it left off.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests

import config

_state = {"out_dir": Path("outputs"), "fresh": False, "last_call": 0.0, "json_mode": config.JSON_MODE,
          "calls_made": 0, "calls_reused": 0, "names_used": set()}


class QuotaExhausted(Exception):
    pass


def load_env(path=".env"):
    """Tiny .env reader (KEY=value lines), so we don't need an extra package."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def settings():
    load_env()
    base = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
    return {
        "base_url": base,
        "api_key": os.environ.get("LLM_API_KEY", ""),
        "model": os.environ.get("LLM_MODEL", "gemini-2.5-flash"),
    }


def configure(out_dir, fresh=False):
    _state["out_dir"] = Path(out_dir)
    _state["fresh"] = fresh
    (_state["out_dir"] / "raw_llm").mkdir(parents=True, exist_ok=True)


def extract_json(text):
    """LLMs sometimes wrap JSON in ```json fences or add a sentence. Pull out the {...} part."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def _post(messages):
    """One HTTP request to the chat API, with waiting + retrying on rate limits."""
    s = settings()
    if not s["api_key"]:
        raise SystemExit("LLM_API_KEY is not set. Copy .env.example to .env and put your key in it.")

    # Stay under the free-tier requests-per-minute limit.
    wait = config.SECONDS_BETWEEN_CALLS - (time.time() - _state["last_call"])
    if wait > 0:
        time.sleep(wait)

    body = {"model": s["model"], "messages": messages, "temperature": config.TEMPERATURE}
    if _state["json_mode"]:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {s['api_key']}", "Content-Type": "application/json"}

    for attempt, pause in enumerate([0] + config.RETRY_WAITS):
        if pause:
            print(f"    waiting {pause}s before retry...")
            time.sleep(pause)
        try:
            r = requests.post(f"{s['base_url']}/chat/completions", json=body, headers=headers, timeout=180)
        except requests.RequestException as e:
            print(f"    network error: {e}")
            continue
        _state["last_call"] = time.time()

        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"] or "", data.get("usage", {}) or {}, s["model"]

        # Some providers don't support JSON mode -> switch it off and retry immediately.
        if r.status_code == 400 and _state["json_mode"] and "response_format" in r.text:
            print("    provider does not support JSON mode, retrying without it")
            _state["json_mode"] = False
            body.pop("response_format", None)
            return _post(messages)

        if r.status_code == 429 and re.search(r"per ?day|PerDay|daily", r.text, re.IGNORECASE) and attempt >= 1:
            raise QuotaExhausted(r.text[:300])
        if r.status_code in (429, 500, 502, 503, 504):
            print(f"    HTTP {r.status_code} (rate limit or server busy)")
            continue
        raise SystemExit(f"LLM API error {r.status_code}: {r.text[:500]}")
    raise QuotaExhausted("gave up after several retries (rate limit)")


def ask(name, system, user):
    """
    Ask the model and return the parsed JSON answer (a dict).
    `name` identifies the call (e.g. 'find_w03') and is the file name of the saved answer.
    """
    path = _state["out_dir"] / "raw_llm" / f"{name}.json"
    _state["names_used"].add(name)
    if path.exists() and not _state["fresh"]:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("parsed_ok"):
            _state["calls_reused"] += 1
            return saved["parsed"]

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    record = {"name": name, "time": datetime.now().isoformat(timespec="seconds"),
              "prompt_chars": len(system) + len(user), "attempts": []}
    parsed, ok = None, False

    for attempt in range(2):  # first try + one retry if the JSON is broken
        text, usage, model = _post(messages)
        _state["calls_made"] += 1
        record["model"] = model
        record["attempts"].append({"response_text": text, "usage": usage})
        try:
            parsed, ok = extract_json(text), True
            break
        except (ValueError, json.JSONDecodeError) as e:
            print(f"    {name}: answer was not valid JSON ({e}); asking once more")
            messages = messages + [{"role": "assistant", "content": text},
                                   {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object."}]

    record["parsed_ok"], record["parsed"] = ok, parsed
    # Note: we save the model's answer, not the prompt, because the prompt contains the
    # confidential screenplay text.
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    if not ok:
        print(f"    {name}: still not valid JSON, skipping this call (saved in raw_llm/)")
        return {}
    return parsed


def usage_summary(out_dir, names=None):
    """Add up tokens over the saved answers (used for the cost table). `names` = only these calls."""
    rows = []
    for f in sorted((Path(out_dir) / "raw_llm").glob("*.json")):
        if names is not None and f.stem not in names:
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        tin = sum(a["usage"].get("prompt_tokens", 0) or 0 for a in rec["attempts"])
        tout = sum(a["usage"].get("completion_tokens", 0) or 0 for a in rec["attempts"])
        rows.append({"call": rec["name"], "step": rec["name"].split("_")[0], "model": rec.get("model", ""),
                     "attempts": len(rec["attempts"]), "tokens_in": tin, "tokens_out": tout,
                     "parsed_ok": rec["parsed_ok"]})
    return rows
