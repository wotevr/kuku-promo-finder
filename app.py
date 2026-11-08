"""
Web version of the promo finder (Streamlit). It runs the same run.py pipeline.

    streamlit run app.py

Nothing is stored: the uploaded scripts and the API key live only in this session's temporary folder.
"""

import io
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
GEMINI_CHAIN = "gemini-3.6-flash,gemini-3.5-flash,gemini-3.7-flash,gemini-flash-lite-latest,gemini-3.5-flash-lite,gemini-flash-latest,gemini-3.8-flash"


def demo_key():
    """Built-in free Gemini key, stored as a Streamlit secret (never shown to visitors)."""
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:  # no secrets file (e.g. running locally)
        return os.environ.get("GEMINI_API_KEY", "")

st.set_page_config(page_title="Kuku Promo Finder", page_icon="🎬", layout="wide")
st.title("🎬 Kuku Promo Finder")
st.write(
    "Finds the best **20-60 second promo clips** in a microdrama screenplay: clips that open on a hook in the "
    "first 3 seconds, end on a cliffhanger, and make sense to someone who has never seen the show. "
    "For each promo you get the episode(s), exact start/end times, the hook, the cliffhanger, scores and a ranked "
    "posting order. Code: [github.com/wotevr/kuku-promo-finder](https://github.com/wotevr/kuku-promo-finder)"
)
st.caption("Your files and API key are used only for this run and are not stored.")

with st.expander("How it works", expanded=False):
    st.markdown(
        "1. **Read** (code): parse every episode into timed lines\n"
        "2. **Find** (LLM): read 4 episodes at a time, propose promos with timestamps\n"
        "3. **Check** (code): reject invented times / wrong length, never cut a line in half, remove duplicates\n"
        "4. **Judge** (LLM): a 'cold viewer' sees only the clip and scores hook, cliffhanger, makes-sense-alone\n"
        "5. **Rank** (code): score formula + variety rule, then one line on why each top clip should be posted"
    )

PRESETS = {
    "Google Gemini": ("gemini", "https://generativelanguage.googleapis.com/v1beta", "gemini-3.6-flash"),
    "OpenAI (GPT)": ("openai", "https://api.openai.com/v1", ""),
    "Anthropic (Claude)": ("openai", "https://api.anthropic.com/v1", "claude-sonnet-5-5"),
    "Groq": ("openai", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "OpenRouter": ("openai", "https://openrouter.ai/api/v1", ""),
    "Other (OpenAI-compatible)": ("openai", "", ""),
}

col1, col2 = st.columns(2)
with col1:
    uploads = st.file_uploader(
        "1. Upload the episode screenplays (the episode-XXX-*.txt files, or one .zip of them)",
        type=["txt", "zip"], accept_multiple_files=True)
    episodes = st.text_input("Episodes to run (e.g. 1-4). Leave empty for all.", value="1-4",
                             help="About 1 minute per 2 episodes on a free key. 1-4 is a good quick test.")
    paid = st.checkbox("This is a paid / high-limit key (skip the free-tier waits between calls)")
with col2:
    preset = st.selectbox("2. Model provider", list(PRESETS))
    provider, default_url, default_model = PRESETS[preset]
    model = st.text_input("Model name", value=default_model, key=f"model_{preset}",
                          help="Type the exact model name the provider gives you.")
    has_demo = provider == "gemini" and bool(demo_key())
    key = st.text_input("API key" + (" (optional: leave empty to use the built-in free key)" if has_demo else ""),
                        type="password", help="Free Gemini key: https://aistudio.google.com/apikey")
    auto_switch = provider == "gemini" and st.checkbox(
        "If this model is busy or out of quota, automatically try other Gemini models", value=True)
    base_url = st.text_input("Base URL (filled in for you)", value=default_url, key=f"url_{preset}")

use_key = key.strip() or (demo_key() if provider == "gemini" else "")
run = st.button("Find promos", type="primary", disabled=not (uploads and use_key and model and base_url))
if not (uploads and use_key and model and base_url):
    st.info("Upload the episode files (or the .zip) to start." + ("" if use_key else " Then paste an API key."))
elif not key.strip():
    st.caption("API key left empty: the built-in free Gemini key will be used. It has a small free daily quota "
               "(it resets every day). If that quota is used up, paste your own key above and run again.")


def save_uploads(files, folder):
    """Save .txt files (and the .txt files inside any .zip) into folder. Returns how many were saved."""
    n = 0
    for f in files:
        data = f.getvalue()
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for member in z.namelist():
                    name = Path(member).name  # drop any folder path from the zip
                    if name.lower().endswith(".txt") and name.lower().startswith("episode-"):
                        (folder / name).write_bytes(z.read(member))
                        n += 1
        elif f.name.lower().startswith("episode-"):
            (folder / Path(f.name).name).write_bytes(data)
            n += 1
    return n


if run:
    tmp = Path(tempfile.mkdtemp(prefix="kuku_"))
    data, out = tmp / "data", tmp / "out"
    data.mkdir()
    n = save_uploads(uploads, data)
    if n == 0:
        st.error("No episode-XXX-*.txt files found in the upload.")
        st.stop()
    st.write(f"Saved {n} episode files. Running...")

    env = {**os.environ, "LLM_PROVIDER": provider, "LLM_API_KEY": use_key, "LLM_MODEL": model.strip(),
           "LLM_BASE_URL": base_url.strip(), "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
           "SECONDS_BETWEEN_CALLS": "0" if paid else "7"}
    if auto_switch:  # try the chosen model first, then the others (each Gemini model has its own free quota)
        chain = [model.strip()] + [m for m in GEMINI_CHAIN.split(",") if m != model.strip()]
        env.update({"LLM_FALLBACK_MODELS": ",".join(chain), "RETRY_WAITS": "8,15"})
    cmd = [sys.executable, str(HERE / "run.py"), "--input", str(data), "--out", str(out)]
    if episodes.strip():
        cmd += ["--episodes", episodes.strip()]

    stages = [("STEP 1", 0.05, "Reading the scripts"), ("STEP 2", 0.10, "Finding candidate promos (LLM)"),
              ("STEP 3", 0.55, "Checking timestamps and removing duplicates"),
              ("STEP 4", 0.60, "Cold-viewer judge is scoring each clip (LLM)"),
              ("STEP 5", 0.90, "Ranking"), ("Done.", 1.0, "Done")]
    bar = st.progress(0.0, text="Starting...")
    st.caption("Each model call can take 20-60 seconds. Keep this tab open until it finishes.")
    log_box = st.empty()
    lines = []
    proc = subprocess.Popen(cmd, cwd=tmp, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        lines.append(line.rstrip())
        for tag, frac, label in stages:
            if line.strip().startswith(tag):
                bar.progress(frac, text=label)
        if "candidates" in line and line.strip().startswith("episodes"):
            bar.progress(min(0.5, 0.1 + 0.05 * sum("candidates" in x for x in lines)), text="Finding: " + line.strip())
        if line.strip().startswith("judged"):
            bar.progress(0.75, text="Judging: " + line.strip())
        log_box.code("\n".join(lines[-25:]))
    proc.wait()
    bar.progress(1.0, text="Finished")
    st.session_state["out"] = str(out)
    st.session_state["log"] = "\n".join(lines)

if "out" in st.session_state:
    out = Path(st.session_state["out"])
    if not (out / "promos.json").exists():
        log = st.session_state["log"]
        if "quota" in log.lower() or "429" in log:
            st.error("The free daily quota of this API key is used up (or the model is rate-limited). "
                     "Paste your own API key in the 'API key' box (a free Gemini key: aistudio.google.com/apikey, "
                     "or any provider from the list) and click Find promos again.")
        elif "401" in log or "403" in log or "API key" in log:
            st.error("The API key was rejected. Check the key and that it belongs to the chosen provider.")
        elif "404" in log:
            st.error("Model not found. Check the exact model name for this provider.")
        else:
            st.error("The run did not finish. See the log below.")
        st.code(st.session_state["log"][-4000:])
        st.stop()

    promos = pd.read_json(out / "promos.json")
    funnel = pd.read_csv(out / "funnel.csv")
    proposed = int(funnel.loc[funnel.stage.str.startswith("candidates proposed"), "count"].sum()) if len(funnel) else 0
    if promos.empty and proposed == 0:
        st.warning("The model's answers could not be used (see the run log below). Click Find promos again - "
                   "free models are sometimes busy - or choose another model or paste your own key.")
        with st.expander("Run log", expanded=True):
            st.code(st.session_state["log"][-5000:])
        st.stop()
    if promos.empty:
        st.warning("The run finished, but no promo passed the checks (see the funnel and the rejection reasons "
                   "below). This usually means the model is too weak to follow the timestamp rules: try a stronger "
                   "model (e.g. gemini-3.6-flash or gemini-3.5-flash) or more episodes.")
        st.dataframe(funnel, hide_index=True, width="stretch")
        rejected = out / "rejected.csv"
        if rejected.exists():
            st.subheader("Rejected candidates")
            st.dataframe(pd.read_csv(rejected), hide_index=True, width="stretch")
        with st.expander("Run log"):
            st.code(st.session_state["log"])
        st.stop()
    st.success(f"Done: {len(promos)} promos ranked.")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Funnel")
        st.dataframe(funnel, hide_index=True, width="stretch")
    with c2:
        st.subheader("Top 10")
        cols = [c for c in ["rank", "segments_text", "duration", "final_score", "moment_type", "hook", "cliffhanger"]
                if c in promos.columns]
        top = promos.head(10)[cols]
        st.dataframe(top, hide_index=True, width="stretch")

    st.subheader("Ranked promos (full detail)")
    st.markdown((out / "ranked_promos.md").read_text(encoding="utf-8"))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out))
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download ranked_promos.md", (out / "ranked_promos.md").read_bytes(), "ranked_promos.md")
    d2.download_button("Download promos.csv", (out / "promos.csv").read_bytes(), "promos.csv")
    d3.download_button("Download all outputs (.zip)", buf.getvalue(), "promo_outputs.zip")
    with st.expander("Run log"):
        st.code(st.session_state["log"])
