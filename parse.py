"""
STEP 1 - READ (plain code, no AI)

Turns each episode screenplay (.txt) into a table with one row per block:

    episode | line_id | kind | speaker | emotion | start | end | text | scene | scene_heading

kind is one of: dialogue, action, camera, sfx
start / end are in SECONDS from the start of that episode.

The screenplay format looks like this:

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    SCENE 01
    Scene heading: INT. CATHEDRAL - DAY
    **MATRIARCH (Cold, Commanding):**
    Start: (00:11.30) | End: (00:13.10)
    "You should turn around, Juliana."
"""

import re
from pathlib import Path

import pandas as pd

# A block header looks like  **CAMERA:**   or   **LADY JULIANA (Confused, Breathless):**
HEADER_RE = re.compile(r"^\*\*(?P<label>.+?):\*\*\s*$")
# A timing line looks like   Start: (00:11.30) | End: (00:13.10)
TIME_RE = re.compile(r"^Start:\s*\((?P<s>\d+:\d+(?:\.\d+)?)\)\s*\|\s*End:\s*\((?P<e>\d+:\d+(?:\.\d+)?)\)")
FILE_EP_RE = re.compile(r"episode-(\d+)", re.IGNORECASE)

NON_DIALOGUE_KINDS = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx"}


def to_seconds(mmss: str) -> float:
    """'01:23.50' -> 83.5"""
    minutes, seconds = mmss.split(":")
    return round(int(minutes) * 60 + float(seconds), 2)


def to_mmss(seconds: float) -> str:
    """83.5 -> '01:23.50'"""
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def parse_episode(path: Path):
    """Parse one episode file. Returns (info_dict, list_of_block_dicts)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Episode number: trust the file name (episode-007-...), it is always present.
    match = FILE_EP_RE.search(path.name)
    episode = int(match.group(1)) if match else None

    def header_field(name):
        for line in lines[:10]:
            if line.startswith(name + ":"):
                return line.split(":", 1)[1].strip()
        return ""

    info = {
        "episode": episode,
        "logline": header_field("EPISODE LOGLINE"),
        "continuity": header_field("CONTINUITY NOTES"),
        "file": path.name,
    }

    blocks = []
    scene_no, scene_heading = 0, ""
    current = None  # the block we are currently filling with text

    for raw in lines:
        line = raw.strip()

        if line.startswith("SCENE ") and line[6:].strip().isdigit():
            scene_no = int(line[6:].strip())
            continue
        if line.startswith("Scene heading:"):
            scene_heading = line.split(":", 1)[1].strip()
            continue

        header = HEADER_RE.match(line)
        if header:
            label = header.group("label").strip()
            if label.upper() in NON_DIALOGUE_KINDS:
                kind, speaker, emotion = NON_DIALOGUE_KINDS[label.upper()], "", ""
            else:
                kind = "dialogue"
                # "LADY JULIANA (Confused, Breathless)" -> speaker + emotion
                m = re.match(r"^(?P<who>[^(]+?)\s*(?:\((?P<emo>.*)\))?$", label)
                speaker = m.group("who").strip() if m else label
                emotion = (m.group("emo") or "").strip() if m else ""
            current = {
                "episode": episode, "kind": kind, "speaker": speaker, "emotion": emotion,
                "start": None, "end": None, "text": "", "scene": scene_no, "scene_heading": scene_heading,
            }
            blocks.append(current)
            continue

        timing = TIME_RE.match(line)
        if timing and current is not None and current["start"] is None:
            current["start"] = to_seconds(timing.group("s"))
            current["end"] = to_seconds(timing.group("e"))
            continue

        # Section labels between scenes are not part of any line.
        if line == "Actions and dialogue:":
            continue
        # Any other non-empty line is the text of the current block.
        if line and current is not None and not line.startswith("---") and not line.startswith("Scene "):
            current["text"] = (current["text"] + " " + line.strip('"')).strip()

    # Drop blocks without a timestamp (should not happen, but be safe).
    blocks = [b for b in blocks if b["start"] is not None]
    for i, b in enumerate(blocks):
        b["line_id"] = f"E{episode:02d}-L{i:03d}"

    info["duration"] = max((b["end"] for b in blocks), default=0.0)
    return info, blocks


def load_show(folder, episodes=None):
    """
    Read every episode file in `folder` (searched recursively).
    `episodes` = optional set of episode numbers to keep (for quick tests).
    Returns (episodes_df, lines_df).
    """
    files = sorted(Path(folder).rglob("episode-*.txt"))
    if not files:
        raise SystemExit(f"No episode-*.txt files found under {folder}")

    infos, all_blocks = [], []
    for path in files:
        info, blocks = parse_episode(path)
        if episodes and info["episode"] not in episodes:
            continue
        infos.append(info)
        all_blocks.extend(blocks)

    lines_df = pd.DataFrame(all_blocks).sort_values(["episode", "start", "end"]).reset_index(drop=True)
    eps_df = pd.DataFrame(infos).sort_values("episode").reset_index(drop=True)

    # Simple per-episode statistics (useful for the write-up).
    counts = lines_df.groupby("episode").agg(
        n_blocks=("kind", "size"),
        n_dialogue=("kind", lambda k: (k == "dialogue").sum()),
        n_action=("kind", lambda k: (k == "action").sum()),
        words=("text", lambda t: sum(len(x.split()) for x in t)),
    )
    eps_df = eps_df.merge(counts, left_on="episode", right_index=True, how="left")
    return eps_df, lines_df


def episode_text_for_llm(lines_df, episode, include_camera=False):
    """
    Turn one episode into compact text for the LLM, one line per block, e.g.
        [E07 00:11.30-00:13.10] MATRIARCH (Cold): "You should turn around, Juliana."
    Camera directions are left out by default to halve the tokens.
    """
    ep = lines_df[lines_df.episode == episode]
    keep = {"dialogue", "action", "sfx"} | ({"camera"} if include_camera else set())
    out = []
    for row in ep[ep.kind.isin(keep)].itertuples():
        stamp = f"[E{episode:02d} {to_mmss(row.start)}-{to_mmss(row.end)}]"
        if row.kind == "dialogue":
            emo = f" ({row.emotion})" if row.emotion else ""
            out.append(f'{stamp} {row.speaker}{emo}: "{row.text}"')
        else:
            out.append(f"{stamp} {row.kind.upper()}: {row.text}")
    return "\n".join(out)


if __name__ == "__main__":
    # Quick check:  python parse.py ./data
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "./data"
    eps, lines = load_show(folder)
    print(eps[["episode", "duration", "n_blocks", "n_dialogue", "n_action", "words"]].to_string(index=False))
    print(f"\n{len(eps)} episodes, {len(lines)} blocks, total runtime {eps.duration.sum() / 60:.1f} min")
