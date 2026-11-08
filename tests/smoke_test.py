"""
Smoke test: runs the WHOLE pipeline with a FAKE LLM (no API key, no cost).

It only tests the plumbing - parsing, timestamp checks, snapping, de-duplication, ranking and
output files. The fake LLM deliberately returns some BAD candidates (invented time, too short,
a cut in the middle of a sentence, a duplicate, one broken-JSON answer) to prove the checks catch them.
Its "scores" are random, so the ranking it produces means nothing.

    python tests/smoke_test.py --input ./data --episodes 1-8
"""

import json
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import llm  # noqa: E402
import prompts  # noqa: E402
import run  # noqa: E402
from parse import to_mmss  # noqa: E402

random.seed(0)
LINE_RE = re.compile(r"^\[E(\d+) (\d+:\d+\.\d+)-(\d+:\d+\.\d+)\] (.*)$", re.MULTILINE)
_broken_once = {"done": False}


def secs(t):
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def fake_find(user):
    by_ep = {}
    for ep, s, e, txt in LINE_RE.findall(user):
        by_ep.setdefault(int(ep), []).append((secs(s), secs(e), txt))
    eps = sorted(by_ep)
    promos = []

    def seg(ep, a, b):
        return {"episode": ep, "start": to_mmss(a), "end": to_mmss(b)}

    def base(segs, kind):
        return {"segments": segs, "hook": f"fake hook ({kind})", "cliffhanger": f"fake cliffhanger ({kind})",
                "why_it_works": "fake", "moment_type": random.choice(["slap", "reveal", "betrayal", "threat"]),
                "spoiler_risk": random.choice(["low", "low", "medium", "high"])}

    for ep in eps:
        L = by_ep[ep]
        # 1. a valid ~30 s clip starting at a real line
        i = random.randrange(0, max(1, len(L) // 2))
        j = next((k for k in range(i, len(L)) if L[k][1] - L[i][0] >= 30), len(L) - 1)
        promos.append(base([seg(ep, L[i][0], L[j][1])], "good"))
        # 2. starts 0.4 s INSIDE a spoken line -> should be snapped back, not rejected
        d = [x for x in L if '"' in x[2] and x[1] - x[0] > 1.0]
        if d:
            a = d[0]
            k = next((k for k in range(len(L)) if L[k][1] - a[0] >= 25), len(L) - 1)
            promos.append(base([seg(ep, a[0] + 0.4, L[k][1])], "mid-sentence start"))
        # 3. too short
        promos.append(base([seg(ep, L[0][0], L[0][0] + 8)], "too short"))
        # 4. duplicate of #1
        promos.append(base([seg(ep, L[i][0], L[j][1])], "duplicate"))
    # 5. invented timestamp (far from every real line)
    promos.append(base([seg(eps[0], 999, 1030)], "invented"))
    # 6. cross-episode: end of one episode + start of the next
    if len(eps) > 1:
        a, b = by_ep[eps[0]], by_ep[eps[1]]
        promos.append(base([seg(eps[0], a[-6][0], a[-1][1]), seg(eps[1], b[0][0], b[5][1])], "cross-episode"))
    return {"promos": promos}


def fake_post(messages):
    system, user = messages[0]["content"], messages[1]["content"]
    usage = {"prompt_tokens": len(system + user) // 4, "completion_tokens": 400}
    if system == prompts.SUMMARY_SYSTEM:
        out = {"summary": "fake summary", "characters": [{"name": "Juliana", "who": "lead"}],
               "big_reveals": [{"episode": 10, "reveal": "fake reveal"}]}
    elif system == prompts.FIND_SYSTEM:
        if not _broken_once["done"] and len(messages) == 2:
            _broken_once["done"] = True
            return "Sure! Here are the promos: {broken json", usage, "fake-model"
        out = fake_find(user)
    elif system == prompts.JUDGE_SYSTEM:
        ids = re.findall(r"### CLIP (P\d+)", user)
        out = {"clips": [{"clip_id": cid, "who_are_they": "x", "what_is_happening": "x",
                          "what_is_unresolved": "x", "understands_alone": random.random() > 0.2,
                          "hook_strength": random.randint(1, 5), "hook_evidence": "x",
                          "cliffhanger_strength": random.randint(1, 5), "cliffhanger_evidence": "x",
                          "makes_sense_alone": random.randint(1, 5), "sense_evidence": "x"}
                         for cid in ids[:-1] or ids]}  # drop the last clip once -> tests "no answer"
    else:
        ids = re.findall(r"#\d+ (P\d+)", user)
        out = {"why": [{"promo_id": i, "why_post_first": "fake reason"} for i in ids]}
    return "```json\n" + json.dumps(out) + "\n```", usage, "fake-model"


if __name__ == "__main__":
    args = sys.argv[1:]
    out = Path("smoke_out")
    shutil.rmtree(out, ignore_errors=True)
    llm.settings = lambda: {"base_url": "fake://", "api_key": "fake", "model": "fake-model"}
    llm._post = fake_post
    llm.config.SECONDS_BETWEEN_CALLS = 0
    sys.argv = ["run.py", "--out", str(out), "--fresh"] + args
    run.main()

    # ---- assertions
    import pandas as pd
    rej = pd.read_csv(out / "rejected.csv")
    promos = json.loads((out / "promos.json").read_text())
    reasons = " ".join(rej.reason.astype(str))
    assert "invented" in reasons, "invented timestamp was not caught"
    assert "too short" in reasons, "too-short clip was not caught"
    assert "duplicate" in reasons, "duplicate was not caught"
    assert any(p["snapped_by"] > 0 for p in promos), "mid-sentence start was not snapped"
    assert all(20 <= p["duration"] <= 60 for p in promos), "a final promo is outside 20-60 s"
    for f in ["ranked_promos.md", "promos.csv", "funnel.csv", "cost.csv", "run_log.txt", "labels.csv", "my_picks.csv"]:
        assert (out / f).exists(), f"missing {f}"
    print("\nSMOKE TEST PASSED (plumbing only - the fake LLM's scores are random)")
