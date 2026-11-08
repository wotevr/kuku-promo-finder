"""
Run this FIRST:  python check_setup.py
Checks that your .env is set up and the model answers. Costs 1 API request.
"""

import requests

import llm


def list_models(s):
    if s["provider"] == "gemini":
        r = requests.get(f"{s['base_url']}/models", headers={"x-goog-api-key": s["api_key"]},
                         params={"pageSize": 200}, timeout=30)
        if not r.ok:
            return None, r.status_code
        return sorted(m["name"].replace("models/", "") for m in r.json().get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", [])), 200
    r = requests.get(f"{s['base_url']}/models", headers={"Authorization": f"Bearer {s['api_key']}"}, timeout=30)
    if not r.ok:
        return None, r.status_code
    return sorted(m.get("id", "") for m in r.json().get("data", [])), 200


def main():
    s = llm.settings()
    key = s["api_key"]
    print(f"Provider : {s['provider']}")
    print(f"Base URL : {s['base_url']}")
    print(f"Model    : {s['model']}")
    print(f"API key  : {'(missing!)' if not key else key[:4] + '...' + key[-4:]}")
    if not key:
        print("\nCopy .env.example to .env and paste your key after LLM_API_KEY=")
        return

    try:
        ids, status = list_models(s)
        if ids is None:
            print(f"\n(could not list models: HTTP {status}; that is OK for some providers)")
        else:
            print(f"\nModels available to this key ({len(ids)}):")
            print("  " + "\n  ".join(ids[:60]))
            if s["model"] not in ids:
                print(f"\n(note: '{s['model']}' is not in that list; aliases like *-latest may still work)")
    except requests.RequestException as e:
        print(f"\nNetwork problem: {e}")
        return

    try:
        text, usage, model = llm._post([{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}])
        print(f"\nTEST CALL WORKED. {model} replied: {text.strip()[:80]}")
        print("You are ready. Quick test:  python run.py --input ./data --out test_out/ --episodes 1-4")
    except (SystemExit, llm.QuotaExhausted) as e:
        print(f"\nTEST CALL FAILED:\n{e}")


if __name__ == "__main__":
    main()
