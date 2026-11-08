"""
Kuku promo finder - finds 20-60 s promo clips in a microdrama screenplay.

    python run.py --input ./data --out outputs/            # full show
    python run.py --input ./data --out test_out/ --episodes 1-5   # quick test

Steps:
  1 READ   (code)  parse every episode into timed lines
  2 FIND   (LLM)   read a window of episodes, propose candidate promos with timestamps
  3 CHECK  (code)  reject invented / too short / too long clips, snap cuts to line edges, remove duplicates
  4 JUDGE  (LLM)   a "cold viewer" reads ONLY the clip and scores hook / cliffhanger / makes-sense-alone
  5 RANK   (code + 1 LLM call)  score formula + diversity rule, then one line on why to post each top clip
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import checks
import config
import llm
import prompts
from parse import episode_text_for_llm, load_show, to_mmss


# ------------------------------------------------------------------ helpers
def parse_episode_list(text):
    """'1-5' -> {1..5};  '1,4,9' -> {1,4,9}"""
    if not text:
        return None
    eps = set()
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            eps.update(range(int(a), int(b) + 1))
        else:
            eps.add(int(part))
    return eps


def make_windows(episodes, size):
    """Windows of `size` episodes that overlap by one: [1-4], [4-7], [7-10], ..."""
    if len(episodes) <= size:
        return [episodes]
    windows, i = [], 0
    while True:
        windows.append(episodes[i:i + size])
        if i + size >= len(episodes):
            return windows
        i += size - 1


def fingerprint(*parts):
    """Short fingerprint of a question: a saved answer is reused only for the exact same question."""
    return hashlib.md5("\n".join(parts).encode()).hexdigest()[:6]


def ask(prefix, system, user):
    return llm.ask(f"{prefix}_{fingerprint(system, user)}", system, user)


def segs_to_text(segs):
    return " + ".join(f"E{s['episode']:02d} {to_mmss(s['start'])}-{to_mmss(s['end'])}" for s in segs)


def reason_category(reason):
    """'too short (12.3s)' -> 'too short'  (for counting)"""
    for key in ["invented", "too short", "too long", "duplicate", "out of order", "not in the text",
                "outside the episode", "too many segments", "no segments", "unreadable", "judge"]:
        if key in reason:
            return key
    return reason


# ------------------------------------------------------------------ step 2: FIND
def story_context(eps_df, title):
    loglines = "\n".join(f"Episode {r.episode}: {r.logline}" for r in eps_df.itertuples())
    s = ask("summary", prompts.SUMMARY_SYSTEM, prompts.SUMMARY_USER.format(title=title, loglines=loglines))
    return {
        "summary": s.get("summary", ""),
        "characters": "; ".join(f"{c.get('name')}: {c.get('who')}" for c in s.get("characters", [])),
        "reveals": "; ".join(f"Ep {r.get('episode')}: {r.get('reveal')}" for r in s.get("big_reveals", [])),
        "raw": s,
    }


def find_candidates(window, eps_df, lines_df, ctx, title, prefix="find"):
    info = eps_df.set_index("episode")
    blocks = []
    for ep in window:
        text = episode_text_for_llm(lines_df, ep, config.INCLUDE_CAMERA_LINES)
        blocks.append(f"=== EPISODE {ep} (length {to_mmss(info.loc[ep, 'duration'])}) ===\n"
                      f"LOGLINE: {info.loc[ep, 'logline']}\n{text}")
    user = prompts.FIND_USER.format(
        title=title, summary=ctx["summary"], characters=ctx["characters"], reveals=ctx["reveals"],
        max_segments=config.MAX_SEGMENTS, min_s=config.MIN_PROMO_SECONDS, max_s=config.MAX_PROMO_SECONDS,
        max_promos=config.MAX_PROMOS_PER_WINDOW, episodes_text="\n\n".join(blocks))
    answer = ask(f"{prefix}_e{window[0]:02d}-{window[-1]:02d}", prompts.FIND_SYSTEM, user)
    return answer.get("promos", []) if isinstance(answer, dict) else []


# ------------------------------------------------------------------ step 4: JUDGE
def judge(promos, lines_df):
    results = {}
    for i in range(0, len(promos), config.CLIPS_PER_JUDGE_CALL):
        batch = promos[i:i + config.CLIPS_PER_JUDGE_CALL]
        clips = "\n\n".join(f"### CLIP {p['promo_id']}\n{checks.clip_text(p['segs'], lines_df)}" for p in batch)
        answer = ask(f"judge_{batch[0]['promo_id']}-{batch[-1]['promo_id']}",
                     prompts.JUDGE_SYSTEM, prompts.JUDGE_USER.format(n=len(batch), clips=clips))
        for c in answer.get("clips", []) if isinstance(answer, dict) else []:
            results[str(c.get("clip_id", "")).strip()] = c
        print(f"  judged {min(i + len(batch), len(promos))}/{len(promos)}")
    return results


def score_1_to_5(value):
    try:
        return float(min(5, max(1, int(value))))
    except (TypeError, ValueError):
        return np.nan


# ------------------------------------------------------------------ step 5: RANK
def rank(promos):
    """Score formula + simple diversity rule for the top N. Returns promos sorted by rank."""
    for p in promos:
        base = sum(config.WEIGHTS[k] * p[k] for k in config.WEIGHTS)
        p["penalty"] = (config.PENALTY_NEEDS_SETUP if p["needs_setup"] else 0.0) \
            + config.PENALTY_SPOILER.get(p["spoiler_risk"], 0.0)
        p["final_score"] = round(base - p["penalty"], 2)

    # Highest score first; ties -> earlier episode first (new viewers start at episode 1), then shorter clip.
    ordered = sorted(promos, key=lambda p: (-p["final_score"], p["segs"][0]["episode"], p["duration"]))

    top, rest, type_count = [], [], Counter()
    for p in ordered:
        if len(top) >= config.TOP_N:
            rest.append(p)
            continue
        same_minute = any(
            q["segs"][0]["episode"] == p["segs"][0]["episode"]
            and abs(q["segs"][0]["start"] - p["segs"][0]["start"]) < config.SAME_MINUTE_SECONDS for q in top)
        too_many_of_type = type_count[p["moment_type"]] >= config.MAX_SAME_TYPE_IN_TOP
        if same_minute or too_many_of_type:
            p["diversity_note"] = "ranked lower for variety: " + (
                "starts within a minute of a higher clip" if same_minute
                else f"top list already has {config.MAX_SAME_TYPE_IN_TOP} '{p['moment_type']}' clips")
            rest.append(p)
        else:
            top.append(p)
            type_count[p["moment_type"]] += 1

    final = top + rest
    for i, p in enumerate(final, 1):
        p["rank"] = i
    return final


def why_post_first(top, title):
    lines = []
    for p in top:
        lines.append(f"#{p['rank']} {p['promo_id']} | {segs_to_text(p['segs'])} | {p['moment_type']} | "
                     f"hook {p['hook_strength']:.0f}, cliffhanger {p['cliffhanger_strength']:.0f}, "
                     f"alone {p['makes_sense_alone']:.0f}\n   hook: {p['hook']}\n   cliffhanger: {p['cliffhanger']}")
    answer = ask("why", prompts.WHY_SYSTEM, prompts.WHY_USER.format(n=len(top), title=title, promos="\n".join(lines)))
    return {str(w.get("promo_id")): w.get("why_post_first", "") for w in answer.get("why", [])} if answer else {}


# ------------------------------------------------------------------ outputs
def write_markdown(final, path, title, model):
    md = [f"# Ranked promos - {title}", "",
          f"Model: `{model}`. Scores are 1-5 from the cold-viewer judge. "
          f"final = hook + cliffhanger + makes-sense-alone - penalties "
          f"(needs setup -{config.PENALTY_NEEDS_SETUP:g}, spoiler medium -1 / high -2). "
          f"Times are mm:ss within each episode.", ""]
    for p in final:
        if p["rank"] == config.TOP_N + 1:
            md += ["---", "", "## The rest (ranked)", ""]
        md += [f"### #{p['rank']} - {p['promo_id']} - {p['duration']:.1f}s - score {p['final_score']:g} "
               f"(hook {p['hook_strength']:.0f} / cliffhanger {p['cliffhanger_strength']:.0f} / "
               f"alone {p['makes_sense_alone']:.0f})",
               f"- **Segments:** {segs_to_text(p['segs'])}",
               f"- **Hook:** {p['hook']}",
               f"- **Plays in first 3 s:** {p['first_3s'] or '(no dialogue/action line starts in the first 3 s)'}",
               f"- **Cliffhanger:** {p['cliffhanger']}",
               f"- **Last line:** {p['last_line']}",
               f"- **Why it works:** {p['why_it_works']}"]
        if p.get("why_post_first"):
            md.append(f"- **Why post it at #{p['rank']}:** {p['why_post_first']}")
        md.append(f"- **Cold viewer:** {'understands it' if not p['needs_setup'] else 'NEEDS SETUP'} - "
                  f"unresolved: {p['what_is_unresolved']}")
        flags = [f"type: {p['moment_type']}", f"spoiler risk: {p['spoiler_risk']}"]
        if p["snapped_by"] > 0.3:
            flags.append(f"cut moved {p['snapped_by']:.1f}s to avoid splitting a line")
        if p.get("diversity_note"):
            flags.append(p["diversity_note"])
        md += [f"- *{'; '.join(flags)}*", ""]
    Path(path).write_text("\n".join(md), encoding="utf-8")


def public_record(p):
    """Promo as a plain dict for promos.json / promos.csv."""
    keep = ["rank", "promo_id", "window", "duration", "final_score", "hook_strength", "cliffhanger_strength",
            "makes_sense_alone", "penalty", "needs_setup", "spoiler_risk", "moment_type", "hook", "cliffhanger",
            "why_it_works", "why_post_first", "first_3s", "last_line", "who_are_they", "what_is_happening",
            "what_is_unresolved", "hook_evidence", "cliffhanger_evidence", "sense_evidence", "snapped_by",
            "diversity_note"]
    rec = {k: p.get(k, "") for k in keep}
    rec["segments"] = [{"episode": s["episode"], "start": to_mmss(s["start"]), "end": to_mmss(s["end"])}
                       for s in p["segs"]]
    rec["segments_text"] = segs_to_text(p["segs"])
    rec["episodes"] = ",".join(str(e) for e in sorted({s["episode"] for s in p["segs"]}))
    return rec


def write_label_templates(final, out, ep_list):
    """Files for the HUMAN check. Never overwrite them - they may already contain my ratings."""
    labels = out / "labels.csv"
    if not labels.exists():
        rows = [{"rank": p["rank"], "promo_id": p["promo_id"], "segments": segs_to_text(p["segs"]),
                 "first_3s": p["first_3s"], "last_line": p["last_line"], "hook": p["hook"],
                 "cliffhanger": p["cliffhanger"], "my_rating": "", "my_note": ""} for p in final[:20]]
        pd.DataFrame(rows).to_csv(labels, index=False, encoding="utf-8-sig")  # -sig so Excel shows it right
    picks = out / "my_picks.csv"
    if not picks.exists():
        chosen = sorted({ep_list[int(i)] for i in np.linspace(0, len(ep_list) - 1, min(6, len(ep_list)))})
        pd.DataFrame([{"episode": e, "start": "", "end": "", "note": ""} for e in chosen]).to_csv(picks, index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Find promo clips in a microdrama screenplay.")
    ap.add_argument("--input", default="./data", help="folder with episode-XXX-*.txt files")
    ap.add_argument("--out", default="outputs", help="output folder")
    ap.add_argument("--episodes", default="", help="e.g. 1-5 or 1,3,7 (default: all)")
    ap.add_argument("--title", default="", help="show title (default: name of the episode folder)")
    ap.add_argument("--fresh", action="store_true", help="ignore saved LLM answers and ask again")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    llm.configure(out, args.fresh)
    model = llm.settings()["model"]

    # ---- STEP 1: READ
    eps_df, lines_df = load_show(args.input, parse_episode_list(args.episodes))
    title = args.title or next(Path(args.input).rglob("episode-*.txt")).parent.name
    durations = dict(zip(eps_df.episode, eps_df.duration))
    ep_list = sorted(durations)
    eps_df.to_csv(out / "episodes.csv", index=False)
    print(f"STEP 1 READ: {len(ep_list)} episodes, {len(lines_df)} timed lines, "
          f"{eps_df.duration.sum() / 60:.1f} min of story. Model: {model}")

    # ---- STEP 2: FIND
    ctx = story_context(eps_df, title)
    (out / "story_summary.json").write_text(json.dumps(ctx["raw"], indent=2, ensure_ascii=False), encoding="utf-8")
    windows = make_windows(ep_list, config.EPISODES_PER_WINDOW)
    print(f"STEP 2 FIND: {len(windows)} windows of up to {config.EPISODES_PER_WINDOW} episodes")

    all_cands, rejected, valid = [], [], []
    for w in windows:
        cands = find_candidates(w, eps_df, lines_df, ctx, title)
        print(f"  episodes {w[0]}-{w[-1]}: {len(cands)} candidates")
        for c in cands:
            if not isinstance(c, dict):
                continue
            pid = f"P{len(all_cands) + 1:03d}"
            win = f"{w[0]}-{w[-1]}"
            raw_segs = json.dumps(c.get("segments"), ensure_ascii=False)
            all_cands.append({"promo_id": pid, "window": win, "raw_segments": raw_segs,
                              "hook": c.get("hook", ""), "cliffhanger": c.get("cliffhanger", "")})
            # ---- STEP 3: CHECK
            segs, reason = checks.check_candidate(c, set(w), lines_df, durations)
            if reason:
                rejected.append({"promo_id": pid, "window": win, "stage": "check", "reason": reason,
                                 "raw_segments": raw_segs, "hook": c.get("hook", "")})
                continue
            first3, last = checks.first_and_last(segs, lines_df)
            valid.append({
                "promo_id": pid, "window": win, "segs": segs,
                "duration": round(sum(s["end"] - s["start"] for s in segs), 1),
                "snapped_by": round(sum(s["snapped_by"] for s in segs), 2),
                "hook": c.get("hook", ""), "cliffhanger": c.get("cliffhanger", ""),
                "why_it_works": c.get("why_it_works", ""),
                "moment_type": str(c.get("moment_type", "other")).lower().split("|")[0].strip() or "other",
                "spoiler_risk": str(c.get("spoiler_risk", "low")).lower().strip()
                if str(c.get("spoiler_risk", "low")).lower().strip() in config.PENALTY_SPOILER else "low",
                "first_3s": first3, "last_line": last})

    pd.DataFrame(all_cands).to_csv(out / "candidates_all.csv", index=False)
    pd.DataFrame([{**{k: v for k, v in p.items() if k != "segs"}, "segments": segs_to_text(p["segs"])}
                  for p in valid]).to_csv(out / "candidates_valid.csv", index=False)

    # ---- STEP 3b: remove duplicates (overlapping windows find the same moment twice)
    unique = []
    for p in valid:
        dup = next((q for q in unique if checks.overlap_share(p["segs"], q["segs"]) >= config.DUPLICATE_OVERLAP), None)
        if dup:
            rejected.append({"promo_id": p["promo_id"], "window": p["window"], "stage": "dedupe",
                             "reason": f"duplicate of {dup['promo_id']}", "raw_segments": segs_to_text(p["segs"]),
                             "hook": p["hook"]})
        else:
            unique.append(p)
    print(f"STEP 3 CHECK: {len(all_cands)} found -> {len(valid)} valid -> {len(unique)} unique")

    # ---- STEP 4: JUDGE
    print("STEP 4 JUDGE (cold viewer + scores)")
    verdicts = judge(unique, lines_df)
    judged = []
    for p in unique:
        v = verdicts.get(p["promo_id"])
        if not v:
            rejected.append({"promo_id": p["promo_id"], "window": p["window"], "stage": "judge",
                             "reason": "judge returned no answer for this clip",
                             "raw_segments": segs_to_text(p["segs"]), "hook": p["hook"]})
            continue
        p.update({k: v.get(k, "") for k in ["who_are_they", "what_is_happening", "what_is_unresolved",
                                            "hook_evidence", "cliffhanger_evidence", "sense_evidence"]})
        for k in ["hook_strength", "cliffhanger_strength", "makes_sense_alone"]:
            p[k] = score_1_to_5(v.get(k))
        if any(np.isnan(p[k]) for k in config.WEIGHTS):
            rejected.append({"promo_id": p["promo_id"], "window": p["window"], "stage": "judge",
                             "reason": "judge gave unreadable scores", "raw_segments": segs_to_text(p["segs"]),
                             "hook": p["hook"]})
            continue
        p["needs_setup"] = str(v.get("understands_alone", True)).lower() in ("false", "no", "0")
        judged.append(p)

    # ---- STEP 5: RANK
    final = rank(judged)
    top = final[:config.TOP_N]
    if top:
        why = why_post_first(top, title)
        for p in top:
            p["why_post_first"] = why.get(p["promo_id"], "")
    print(f"STEP 5 RANK: {len(final)} promos ranked")

    # ---- write outputs
    records = [public_record(p) for p in final]
    (out / "promos.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k != "segments"} for r in records]).to_csv(out / "promos.csv", index=False)
    write_markdown(final, out / "ranked_promos.md", title, model)
    pd.DataFrame(rejected, columns=["promo_id", "window", "stage", "reason", "raw_segments", "hook"]).to_csv(
        out / "rejected.csv", index=False)
    write_label_templates(final, out, ep_list)

    reasons = Counter(reason_category(r["reason"]) for r in rejected)
    funnel = [("candidates proposed by LLM", len(all_cands)),
              ("passed timestamp/length checks", len(valid)),
              ("unique after removing duplicates", len(unique)),
              ("judged by cold viewer", len(judged)),
              ("  of which needs setup (could not follow alone)", sum(p["needs_setup"] for p in judged)),
              ("final ranked promos", len(final))]
    funnel += [(f"rejected: {k}", v) for k, v in reasons.most_common()]
    pd.DataFrame(funnel, columns=["stage", "count"]).to_csv(out / "funnel.csv", index=False)

    # ---- cost + log
    usage = pd.DataFrame(llm.usage_summary(out, llm._state["names_used"]))
    runtime = time.time() - t0
    lines = [f"Show: {title}", f"Model: {model}  ({llm.settings()['base_url']})",
             f"Episodes: {len(ep_list)}  |  runtime this run: {runtime / 60:.1f} min  |  "
             f"API calls this run: {llm._state['calls_made']}  (reused saved answers: {llm._state['calls_reused']})", ""]
    lines += ["FUNNEL"] + [f"  {s:<52} {c}" for s, c in funnel] + [""]
    if not usage.empty:
        step = usage.groupby("step").agg(calls=("call", "size"), tokens_in=("tokens_in", "sum"),
                                         tokens_out=("tokens_out", "sum")).reset_index()
        step["cost_usd"] = (step.tokens_in * config.PRICE_IN_PER_M + step.tokens_out * config.PRICE_OUT_PER_M) / 1e6
        total = step[["calls", "tokens_in", "tokens_out", "cost_usd"]].sum()
        n_eps = len(ep_list)
        scale_eps = config.SCALE_SHOWS * config.SCALE_EPISODES_PER_SHOW
        cost_rows = step.to_dict("records") + [
            {"step": "TOTAL", **total.to_dict()},
            {"step": "per episode", "calls": total.calls / n_eps, "tokens_in": total.tokens_in / n_eps,
             "tokens_out": total.tokens_out / n_eps, "cost_usd": total.cost_usd / n_eps},
            {"step": f"scale: {config.SCALE_SHOWS} shows x {config.SCALE_EPISODES_PER_SHOW} episodes",
             "calls": total.calls / n_eps * scale_eps, "tokens_in": total.tokens_in / n_eps * scale_eps,
             "tokens_out": total.tokens_out / n_eps * scale_eps, "cost_usd": total.cost_usd / n_eps * scale_eps}]
        cost = pd.DataFrame(cost_rows)
        cost.to_csv(out / "cost.csv", index=False)
        models = usage.groupby("model").size()
        lines += ["Models that answered: " + ", ".join(f"{m} ({n} calls)" for m, n in models.items()), ""]
        lines += [f"COST (assumed price ${config.PRICE_IN_PER_M}/M input, ${config.PRICE_OUT_PER_M}/M output tokens; "
                  "the free tier actually cost $0)", cost.round(4).to_string(index=False), ""]
    lines += ["TOP 10"] + [f"  #{p['rank']:<2} {p['promo_id']} {segs_to_text(p['segs']):<40} "
                           f"score {p['final_score']:<5g} {p['moment_type']:<13} {p['hook'][:70]}" for p in top]
    (out / "run_log.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nDone. Open {out / 'ranked_promos.md'}. Then rate the top 20 in {out / 'labels.csv'} "
          f"and add your own picks to {out / 'my_picks.csv'}, then run: python eval.py --out {out}")


if __name__ == "__main__":
    try:
        main()
    except llm.QuotaExhausted as e:
        print("\nSTOPPED: the API's daily/minute quota is used up.\n"
              "Everything answered so far is saved in outputs/raw_llm/ and will be REUSED.\n"
              "Fix: wait and run the same command again, or put another free model/provider in .env "
              "and run again.\nDetails: " + str(e))
