"""
Run this FIRST:  python check_setup.py
Checks that your .env is set up and the model answers. Costs 1 API request.
"""

import requests

import llm


def main():
    s = llm.settings()
    key = s["api_key"]
    print(f"Base URL : {s['base_url']}")
    print(f"Model    : {s['model']}")
    print(f"API key  : {'(missing!)' if not key else key[:4] + '...' + key[-4:]}")
    if not key:
        print("\nCopy .env.example to .env and paste your key after LLM_API_KEY=")
        return

    headers = {"Authorization": f"Bearer {key}"}
    try:
        r = requests.get(f"{s['base_url']}/models", headers=headers, timeout=30)
        if r.ok:
            ids = sorted(m.get("id", "").replace("models/", "") for m in r.json().get("data", []))
            print(f"\nModels available to this key ({len(ids)}):")
            print("  " + "\n  ".join(ids[:60]))
            if s["model"] not in ids:
                print(f"\n!! '{s['model']}' is not in that list - pick one of the names above for LLM_MODEL in .env")
        else:
            print(f"\n(could not list models: HTTP {r.status_code}; that is OK for some providers)")
    except requests.RequestException as e:
        print(f"\nNetwork problem: {e}")
        return

    body = {"model": s["model"], "messages": [{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}]}
    r = requests.post(f"{s['base_url']}/chat/completions", json=body,
                      headers={**headers, "Content-Type": "application/json"}, timeout=60)
    if r.ok:
        print(f"\nTest call worked. The model replied: {r.json()['choices'][0]['message']['content'].strip()[:80]}")
        print("You are ready. Quick test:  python run.py --input ./data --out test_out/ --episodes 1-4")
    else:
        print(f"\nTest call FAILED: HTTP {r.status_code}\n{r.text[:600]}")


if __name__ == "__main__":
    main()
