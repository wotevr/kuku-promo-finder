"""
QUALITY CHECK against MY OWN judgement (the human in the loop).

Before running this, fill in two files that run.py created in the output folder:
  1. labels.csv   - for each of the tool's top 20 promos, write good / ok / bad in "my_rating"
  2. my_picks.csv - for a few episodes, write the moments I would pick myself (episode, start, end as mm:ss)
                    add more rows if you have several picks for one episode

    python eval.py --out outputs

Prints and saves (eval_results.md):
  - precision@10 / @20: share of the tool's top picks I rated good (strict) or good+ok (lenient)
  - recall: share of MY picks that the tool also found (any final promo overlapping >= 30% of my pick)
  - Spearman correlation between the tool's score and my rating
"""

import argparse
from pathlib import Path

import pandas as pd

from checks import parse_time

RATING_VALUE = {"good": 2, "ok": 1, "bad": 0}


def overlap_seconds(ep, a, b, segments):
    return sum(max(0.0, min(b, parse_time(s["end"])) - max(a, parse_time(s["start"])))
               for s in segments if int(s["episode"]) == ep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    out = Path(ap.parse_args().out)
    promos = pd.read_json(out / "promos.json", encoding="utf-8")
    report = ["# Evaluation against my own labels", ""]

    # ---------------- precision: how many of the tool's top picks are actually good?
    labels = pd.read_csv(out / "labels.csv", dtype=str, encoding="utf-8-sig", encoding_errors="replace").fillna("")
    labels["my_rating"] = labels["my_rating"].str.strip().str.lower()
    rated = labels[labels.my_rating.isin(RATING_VALUE)].copy()
    if rated.empty:
        report += ["**labels not filled yet** - write good / ok / bad in the my_rating column of labels.csv", ""]
    else:
        rated["rank"] = rated["rank"].astype(int)
        rated["value"] = rated.my_rating.map(RATING_VALUE)
        rated = rated.merge(promos[["promo_id", "final_score"]], on="promo_id", how="left")
        report += [f"Rated promos: {len(rated)} of {len(labels)}", "", "| metric | value |", "|---|---|"]
        for k in (10, 20):
            top = rated[rated["rank"] <= k]
            if len(top):
                strict = (top.my_rating == "good").mean()
                lenient = top.my_rating.isin(["good", "ok"]).mean()
                report.append(f"| precision@{k} strict (good) | {strict:.0%} ({(top.my_rating == 'good').sum()}/{len(top)}) |")
                report.append(f"| precision@{k} lenient (good or ok) | {lenient:.0%} |")
        if rated.value.nunique() > 1 and rated.final_score.nunique() > 1:
            rho = rated["final_score"].corr(rated["value"], method="spearman")
            report.append(f"| Spearman(tool score, my rating) | {rho:.2f} |")
        report.append("")
        bad = rated[rated.my_rating == "bad"]
        if len(bad):
            report += ["Promos I rated BAD (feed these into FAILURES.md):", ""]
            report += [f"- #{r.rank} {r.promo_id} {r.segments}: {r.my_note}" for r in bad.itertuples()]
            report.append("")

    # ---------------- recall: did the tool find the moments I would have picked?
    picks = pd.read_csv(out / "my_picks.csv", dtype=str, encoding="utf-8-sig", encoding_errors="replace").fillna("")
    picks = picks[(picks.start.str.strip() != "") & (picks.end.str.strip() != "")]
    if picks.empty:
        report += ["**my_picks.csv not filled yet** - add the moments you would pick (episode, start, end)", ""]
    else:
        rows, found, found_top20 = [], 0, 0
        for p in picks.itertuples():
            ep, a, b = int(p.episode), parse_time(p.start), parse_time(p.end)
            if a is None or b is None or b <= a:
                print(f"skipping unreadable pick: episode {p.episode} {p.start}-{p.end} (use mm:ss like 01:05.00)")
                continue
            best = max(((overlap_seconds(ep, a, b, r.segments), r) for r in promos.itertuples()),
                       key=lambda x: x[0], default=(0, None))
            share = best[0] / max(1e-9, b - a)
            hit = share >= 0.3
            found += hit
            found_top20 += hit and best[1].rank <= 20
            rows.append(f"| E{ep:02d} {p.start}-{p.end} | {p.note} | "
                        f"{'YES' if hit else 'no'} | {best[1].promo_id if hit else '-'} | "
                        f"{best[1].rank if hit else '-'} | {share:.0%} |")
        n = max(1, len(rows))
        report += [f"Recall: the tool found **{found}/{len(rows)}** of my picks "
                   f"({found / n:.0%}); {found_top20} of them are in its top 20.", "",
                   "| my pick | note | found? | promo | tool rank | overlap |", "|---|---|---|---|---|---|"] + rows

    text = "\n".join(report)
    (out / "eval_results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
