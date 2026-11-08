"""
Helper for the human check: writes the full script of the top promos into one file, so you can
read each clip like a viewer would and rate it in labels.csv.

    python show_clips.py --input ./data --out outputs              # top 20 -> outputs/top_clips_to_review.md
    python show_clips.py --input ./data --out outputs --ids P012 P040
"""

import argparse
import json
from pathlib import Path

from checks import parse_time
from parse import load_show, to_mmss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()
    out = Path(args.out)

    promos = json.loads((out / "promos.json").read_text(encoding="utf-8"))
    chosen = [p for p in promos if p["promo_id"] in args.ids] if args.ids else promos[:args.top]
    _, lines = load_show(args.input, {s["episode"] for p in chosen for s in p["segments"]})

    md = ["# Top promos: full script of each clip (for rating in labels.csv)", ""]
    for p in chosen:
        md += [f"## #{p['rank']} {p['promo_id']} - {p['segments_text']} - {p['duration']}s - score {p['final_score']}",
               f"*Tool says* hook: {p['hook']} | cliffhanger: {p['cliffhanger']}", ""]
        clip_t = 0.0
        for i, s in enumerate(p["segments"]):
            a, b = parse_time(s["start"]), parse_time(s["end"])
            if i:
                md.append("    ------------------------------ CUT ------------------------------")
            ep = lines[(lines.episode == s["episode"]) & (lines.start >= a - 0.01) & (lines.start < b)]
            for r in ep.itertuples():
                t = clip_t + r.start - a
                who = f"{r.speaker} ({r.emotion})" if r.kind == "dialogue" else r.kind.upper()
                md.append(f"    {t:5.1f}s  E{r.episode:02d} {to_mmss(r.start)}  {who}: {r.text}")
            clip_t += b - a
        md += ["", "My rating (good / ok / bad): ____", ""]
    path = out / "top_clips_to_review.md"
    path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {path} ({len(chosen)} clips)")


if __name__ == "__main__":
    main()
