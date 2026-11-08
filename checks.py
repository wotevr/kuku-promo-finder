"""
STEP 3 - CHECK (plain code, no AI)

LLMs are good at judging drama but bad at exact numbers: they invent timestamps, cut lines in
half and miscount durations. Every rule here is a simple, explainable check against the script.
"""

import re

import config
from parse import to_mmss


def parse_time(value):
    """Accept '01:12.40', '1:12.4', '72.4' or 72.4 -> seconds (float). Returns None if unreadable."""
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", value)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    try:
        return float(value)
    except ValueError:
        return None


def _dialogue_spans(lines_df, episode):
    d = lines_df[(lines_df.episode == episode) & (lines_df.kind == "dialogue")]
    return list(zip(d.start, d.end))


def snap(start, end, spans):
    """
    Move cut points so that no line of dialogue is cut in half:
    - if the start falls inside a spoken line, move it back to the start of that line
    - if the end falls inside a spoken line, move it forward to the end of that line
    """
    changed = True
    while changed:
        changed = False
        for a, b in spans:
            if a < start < b:
                start, changed = a, True
            if a < end < b:
                end, changed = b, True
    return round(start, 2), round(end, 2)


def check_candidate(cand, window_eps, lines_df, durations):
    """
    Returns (clean_segments, None) if the candidate passes, or (None, reason) if it is rejected.
    clean_segments = [{"episode", "start", "end", "snapped_by"}] with times in seconds.
    """
    segs = cand.get("segments") or []
    if not isinstance(segs, list) or not segs:
        return None, "no segments"
    if len(segs) > config.MAX_SEGMENTS:
        return None, f"too many segments ({len(segs)})"

    clean = []
    for seg in segs:
        try:
            ep = int(seg.get("episode"))
        except (TypeError, ValueError, AttributeError):
            return None, "segment has no valid episode number"
        if ep not in window_eps:
            return None, f"episode {ep} was not in the text the model saw"
        s, e = parse_time(seg.get("start")), parse_time(seg.get("end"))
        if s is None or e is None:
            return None, "unreadable timestamp"
        if not (0 <= s < e <= durations[ep] + 0.5):
            return None, "timestamp outside the episode / end before start"

        # Does this time really exist in the script? (start of some line / end of some line)
        ep_lines = lines_df[lines_df.episode == ep]
        if (ep_lines.start - s).abs().min() > config.TIMESTAMP_TOLERANCE:
            return None, f"invented start time E{ep:02d} {to_mmss(s)}"
        if (ep_lines.end - e).abs().min() > config.TIMESTAMP_TOLERANCE:
            return None, f"invented end time E{ep:02d} {to_mmss(e)}"

        s2, e2 = snap(s, e, _dialogue_spans(lines_df, ep))
        clean.append({"episode": ep, "start": s2, "end": e2,
                      "snapped_by": round(abs(s2 - s) + abs(e2 - e), 2)})

    # Segments must be in story order and must not overlap.
    for prev, nxt in zip(clean, clean[1:]):
        if (nxt["episode"], nxt["start"]) < (prev["episode"], prev["end"]):
            return None, "segments out of order or overlapping"

    total = sum(s["end"] - s["start"] for s in clean)
    if total < config.MIN_PROMO_SECONDS:
        return None, f"too short ({total:.1f}s)"
    if total > config.MAX_PROMO_SECONDS:
        return None, f"too long ({total:.1f}s)"
    return clean, None


def overlap_share(segs_a, segs_b):
    """Seconds shared by two promos / length of the shorter one (0 = different, 1 = same moment)."""
    shared = 0.0
    for a in segs_a:
        for b in segs_b:
            if a["episode"] == b["episode"]:
                shared += max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    len_a = sum(s["end"] - s["start"] for s in segs_a)
    len_b = sum(s["end"] - s["start"] for s in segs_b)
    return shared / max(1e-9, min(len_a, len_b))


def clip_lines(segs, lines_df):
    """
    The lines a viewer would actually see in the clip, with times measured from the clip start.
    Returns a list of (clip_time_seconds, row) for dialogue / action / sound lines.
    A line is included if it starts inside a segment.
    """
    out, offset = [], 0.0
    for i, seg in enumerate(segs):
        ep = lines_df[(lines_df.episode == seg["episode"]) & lines_df.kind.isin(["dialogue", "action", "sfx"])]
        inside = ep[(ep.start >= seg["start"] - 0.01) & (ep.start < seg["end"])]
        for row in inside.itertuples():
            out.append((round(offset + row.start - seg["start"], 1), row, i))
        offset += seg["end"] - seg["start"]
    return out


def describe_line(row):
    if row.kind == "dialogue":
        emo = f" ({row.emotion})" if row.emotion else ""
        return f'{row.speaker}{emo}: "{row.text}"'
    return f"{row.kind.upper()}: {row.text}"


def clip_text(segs, lines_df):
    """Clip text for the cold-viewer judge, e.g.  '>>> [0.0s] JULIANA: "..."'"""
    text, last_seg = [], 0
    for t, row, seg_i in clip_lines(segs, lines_df):
        if seg_i != last_seg:
            text.append("--- CUT ---")
            last_seg = seg_i
        mark = ">>>" if t < config.HOOK_WINDOW_SECONDS else "   "
        text.append(f"{mark} [{t:.1f}s] {describe_line(row)}")
    return "\n".join(text)


def first_and_last(segs, lines_df):
    """What actually plays in the first 3 seconds, and the last line of the clip (shown to the editor)."""
    lines = clip_lines(segs, lines_df)
    first = [describe_line(r) for t, r, _ in lines if t < config.HOOK_WINDOW_SECONDS]
    last = describe_line(lines[-1][1]) if lines else ""
    return " | ".join(first), last
