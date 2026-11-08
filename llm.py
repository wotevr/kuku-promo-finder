"""
The ONLY file that talks to an AI model.

It speaks two API "dialects": Google's native Gemini API, and the OpenAI-compatible chat API
that Groq, OpenRouter, GitHub Models, OpenAI, Ollama and many others support. So swapping the
model = editing lines in .env (LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL). No code change.

Every answer is saved to outputs/raw_llm/<name>.json. If that file already exists, the saved
answer is reused instead of calling the API again (unless you run with --fresh). This means a
run that stops halfway (e.g. the free daily quota runs out) continues where it left off.
"""

import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests

import config

_state = {"out_dir": Path("outputs"), "fresh": False, "last_call": 0.0, "json_mode": config.JSON_MODE,
          "calls_made": 0, "calls_reused": 0, "names_used": set()}


class QuotaExhausted(Exception):
    pass


_EXHAUSTED_FILE = Path(tempfile.gettempdir()) / "kuku_models_out_of_daily_quota.json"


def _exhausted_today():
    """Models that hit their free DAILY quota in the last 6 hours (shared by all runs on this server)."""
    try:
        data = json.loads(_EXHAUSTED_FILE.read_text())
        return {m for m, t in data.items() if time.time() - t < 6 * 3600}
    except (OSError, ValueError):
        return set()


def _mark_exhausted(model):
    try:
        data = json.loads(_EXHAUSTED_FILE.read_text()) if _EXHAUSTED_FILE.exists() else {}
        data[model] = time.time()
        _EXHAUSTED_FILE.write_text(json.dumps(data))
    except (OSError, ValueError):
        pass


def _post_with_fallback(messages):
    """
    Like _post, but if the model is out of quota / overloaded / not found, switch to the next model in
    LLM_FALLBACK_MODELS (comma-separated) and try again. Free tiers give each model its own daily quota.
    Models known to be out of daily quota are skipped straight away.
    """
    chain = [m.strip() for m in os.environ.get("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
    tried = _state.setdefault("tried_models", set())
    while True:
        current = settings()["model"]
        skip = tried | _exhausted_today()
        if chain and current in skip:
            remaining = [m for m in chain if m not in skip]
            if remaining:
                print(f"    {current} is out of quota today; using {remaining[0]}")
                os.environ["LLM_MODEL"] = remaining[0]
                continue
        try:
            return _post(messages)
        except QuotaExhausted as e:
            tried.add(current)
            if "PerDay" in str(e) or "per day" in str(e).lower():
                _mark_exhausted(current)
            remaining = [m for m in chain if m not in tried | _exhausted_today()]
            if not remaining:
                raise
            print(f"    {current} is unavailable right now ({str(e)[:60]}...); switching to {remaining[0]}")
            os.environ["LLM_MODEL"] = remaining[0]


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
    """
    LLM_PROVIDER = "gemini" -> Google's native Gemini API (works with every AI Studio key type)
    LLM_PROVIDER = "openai" -> any OpenAI-compatible API (Groq, OpenRouter, GitHub Models, OpenAI, Ollama...)
    """
    load_env()
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    default_base = ("https://generativelanguage.googleapis.com/v1beta" if provider == "gemini"
                    else "https://api.openai.com/v1")
    return {
        "provider": provider,
        "base_url": os.environ.get("LLM_BASE_URL", default_base).rstrip("/"),
        "api_key": os.environ.get("LLM_API_KEY", ""),
        "model": os.environ.get("LLM_MODEL", "gemini-3.6-flash"),
    }


def configure(out_dir, fresh=False):
    _state["out_dir"] = Path(out_dir)
    _state["fresh"] = fresh
    (_state["out_dir"] / "raw_llm").mkdir(parents=True, exist_ok=True)


def extract_json(text):
    """
    LLMs sometimes wrap JSON in ```json fences, add a sentence, return a bare list, or put extra text/objects
    after the JSON. Take the first complete JSON value and ignore the rest.
    """
    text = re.sub(r"```(?:json)?", "", text.strip())
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON object found")
    value, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    if isinstance(value, list):  # a bare list: make it fit whichever key the caller expects
        return {"promos": value, "clips": value, "why": value}
    if not isinstance(value, dict):
        raise ValueError("JSON is not an object")
    return value


def _build_request(s, messages):
    """Same question, two API dialects. Returns (url, headers, body)."""
    if s["provider"] == "gemini":
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        gen = {"temperature": config.TEMPERATURE}
        if _state.get("thinking", True) and os.environ.get("LLM_THINKING", "low") != "default":
            gen["thinkingConfig"] = {"thinkingLevel": os.environ.get("LLM_THINKING", "low")}  # faster answers
        if _state["json_mode"]:
            gen["responseMimeType"] = "application/json"
        body = {"contents": contents, "generationConfig": gen,
                # the scripts are fiction with slaps, threats and poison plots: only block truly harmful text
                "safetySettings": [{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in
                                   ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{s['base_url']}/models/{s['model']}:generateContent"
        return url, {"x-goog-api-key": s["api_key"], "Content-Type": "application/json"}, body

    body = {"model": s["model"], "messages": messages, "temperature": config.TEMPERATURE}
    if _state["json_mode"]:
        body["response_format"] = {"type": "json_object"}
    return (f"{s['base_url']}/chat/completions",
            {"Authorization": f"Bearer {s['api_key']}", "Content-Type": "application/json"}, body)


def _read_response(s, data):
    """Pull the answer text and token counts out of either dialect."""
    if s["provider"] == "gemini":
        cands = data.get("candidates") or []
        parts = (cands[0].get("content") or {}).get("parts", []) if cands else []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text:
            reason = cands[0].get("finishReason") if cands else data.get("promptFeedback", {}).get("blockReason")
            print(f"    empty answer from Gemini (reason: {reason})")
        um = data.get("usageMetadata", {})
        usage = {"prompt_tokens": um.get("promptTokenCount", 0),
                 "completion_tokens": um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)}
        return text, usage
    return data["choices"][0]["message"]["content"] or "", data.get("usage", {}) or {}


def _post(messages):
    """One request to the model API, with waiting + retrying on rate limits."""
    s = settings()
    if not s["api_key"]:
        raise SystemExit("LLM_API_KEY is not set. Copy .env.example to .env and put your key in it.")

    # Stay under the free-tier requests-per-minute limit.
    wait = config.SECONDS_BETWEEN_CALLS - (time.time() - _state["last_call"])
    if wait > 0:
        time.sleep(wait)

    url, headers, body = _build_request(s, messages)
    for attempt, pause in enumerate([0] + config.RETRY_WAITS):
        if pause:
            print(f"    waiting {pause}s before retry...")
            time.sleep(pause)
        try:
            r = requests.post(url, json=body, headers=headers,
                              timeout=120 if os.environ.get("LLM_FALLBACK_MODELS") else 300)
        except requests.RequestException as e:
            print(f"    network error: {e}")
            continue
        _state["last_call"] = time.time()

        if r.status_code == 200:
            text, usage = _read_response(s, r.json())
            return text, usage, s["model"]

        # Some models don't accept the "think less" setting -> drop it and retry immediately.
        if r.status_code == 400 and _state.get("thinking", True) and re.search(r"thinking", r.text, re.IGNORECASE):
            print("    model does not accept thinkingLevel, retrying without it")
            _state["thinking"] = False
            return _post(messages)

        # Some providers don't support JSON mode -> switch it off and retry immediately.
        if r.status_code == 400 and _state["json_mode"] and re.search(r"response_format|responseMimeType", r.text):
            print("    provider does not support JSON mode, retrying without it")
            _state["json_mode"] = False
            return _post(messages)

        if r.status_code == 429 and re.search(r"per ?day|PerDay|daily", r.text, re.IGNORECASE) and attempt >= 1:
            raise QuotaExhausted(r.text[:300])
        if r.status_code == 404 and os.environ.get("LLM_FALLBACK_MODELS"):
            raise QuotaExhausted(f"model not found: {r.text[:120]}")
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
        text, usage, model = _post_with_fallback(messages)
        _state["calls_made"] += 1
        record["model"] = model
        record["attempts"].append({"response_text": text, "usage": usage})
        try:
            parsed, ok = extract_json(text), True
            break
        except (ValueError, json.JSONDecodeError) as e:
            print(f"    {name}: answer was not valid JSON ({e}); asking once more")
            if text.strip():  # show the model its broken answer; if it was empty, just ask again
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
