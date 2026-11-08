"""
CONSISTENCY CHECK: if I ask the LLM the exact same question again, does it pick the same promos?

Re-runs STEP 2 (FIND) + STEP 3 (CHECK) for the windows inside --episodes and compares the
valid promos with the ones from the main run (outputs/candidates_valid.csv).
Two promos "match" if they share >= 50% of their time.

    python consistency.py --input ./data --out outputs --episodes 1-7      (2 extra API calls)
"""

import argparse
import re
from pathlib import Path

import pandas as pd

import checks
import config
import llm
import run
from parse import load_show


def parse_segments_text(text):
    """'E01 00:41.70-01:12.10 + E02 00:00.00-00:10.00' -> list of segment dicts (seconds)."""
    segs = []
    for ep, a, b in re.findall(r"E(\d+) (\d+:\d+\.\d+)-(\d+:\d+\.\d+)", str(text)):
        segs.append({"episode": int(ep), "start": checks.parse_time(a), "end": checks.parse_time(b)})
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--episodes", default="1-7")
    args = ap.parse_args()
    out = Path(args.out)
    llm.configure(out)

    eps_df, lines_df = load_show(args.input)  # all episodes, so the story summary is the same as the main run
    title = next(Path(args.input).rglob("episode-*.txt")).parent.name
    durations = dict(zip(eps_df.episode, eps_df.duration))
    ctx = run.story_context(eps_df, title)  # reuses the saved summary answer (no new call)
    wanted = run.parse_episode_list(args.episodes)
    windows = [w for w in run.make_windows(sorted(durations), config.EPISODES_PER_WINDOW) if set(w) <= wanted]
    if not windows:
        raise SystemExit(f"No full window inside episodes {args.episodes}; try a wider range, e.g. 1-7")

    run1 = pd.read_csv(out / "candidates_valid.csv")
    lines, total1, total2, matched1, matched2 = [], 0, 0, 0, 0
    for w in windows:
        win = f"{w[0]}-{w[-1]}"
        first = [parse_segments_text(s) for s in run1[run1.window == win].segments]
        second = []
        for c in run.find_candidates(w, eps_df, lines_df, ctx, title, prefix="consistency"):
            segs, reason = checks.check_candidate(c, set(w), lines_df, durations)
            if segs:
                second.append(segs)
        m1 = sum(any(checks.overlap_share(a, b) >= 0.5 for b in second) for a in first)
        m2 = sum(any(checks.overlap_share(b, a) >= 0.5 for a in first) for b in second)
        total1, total2, matched1, matched2 = total1 + len(first), total2 + len(second), matched1 + m1, matched2 + m2
        lines.append(f"| {win} | {len(first)} | {len(second)} | {m1}/{len(first)} | {m2}/{len(second)} |")

    share1 = matched1 / max(1, total1)
    share2 = matched2 / max(1, total2)
    text = "\n".join([
        "# Consistency: same question asked twice", "",
        f"Model `{llm.settings()['model']}`, temperature {config.TEMPERATURE}. "
        "A promo counts as 'found again' if a promo from the other run shares >= 50% of its time.", "",
        "| window | valid promos run 1 | valid promos run 2 | run-1 promos found again | run-2 promos also in run 1 |",
        "|---|---|---|---|---|", *lines,
        f"| **total** | {total1} | {total2} | **{share1:.0%}** | **{share2:.0%}** |", "",
        "Reading: a low number means the LLM's picks depend on luck -> a human editor should review more than "
        "the top few, or the FIND step should be run twice and the promos found both times trusted more."])
    (out / "consistency.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
