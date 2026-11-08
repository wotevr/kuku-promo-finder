# Kuku Promo Finder

Reads a microdrama screenplay and finds the best **20-60 second promo clips**: ones that open on a
hook in the first 3 seconds, end on a cliffhanger, and make sense to someone who has never seen the
show. For each promo it gives the episode(s), the exact start/end time of every segment, the hook,
the cliffhanger, one line on why it works, and a ranked posting order.

It does **not** train a model. It calls an existing LLM through an API for the judgement calls, and
uses plain Python for everything that must be exact (timestamps, lengths, duplicates).

## How it works (5 steps)

| Step | Who does it | What happens |
|---|---|---|
| 1. READ | code | Parse every episode into a table of timed lines (dialogue, action, sound, camera). |
| 2. FIND | LLM | One call writes a short story summary from the 50 loglines. Then the LLM reads **4 episodes at a time** (windows overlap by 1 episode, so a cliffhanger at the end of one episode and the opening of the next are always seen together) and proposes up to 6 promos per window, with timestamps. |
| 3. CHECK | code | Rejects clips whose timestamps don't exist in the script, that are not 20-60 s, or whose segments are out of order. Moves cut points so no spoken line is cut in half. Removes duplicates (the same moment found by two overlapping windows). Every rejection is logged with its reason. |
| 4. JUDGE | LLM (separate calls) | A "cold viewer" sees **only the clip's own lines** - no summary, no context - and says who these people are, what is happening, what is unresolved, and whether a first-time viewer would follow it. Then it scores hook, cliffhanger and makes-sense-alone from 1 to 5, with one sentence of evidence each. |
| 5. RANK | code + 1 LLM call | final score = hook + cliffhanger + makes-sense-alone (equal weights), minus 2 if the cold viewer could not follow it, minus 1-2 if it gives away a big late twist. The top 10 must be varied (max 3 clips of one type, e.g. slaps; no two starting within the same minute of an episode). One last call writes why each top-10 clip deserves its posting slot. |

Why this split: an LLM is good at judging drama and bad at exact numbers, so the LLM judges and the
code checks. A human editor still approves the final list (see "Quality check").

## Setup (10 minutes)

1. Install Python 3.10+ and then:
   ```
   pip install -r requirements.txt
   ```
2. Get a free API key: Google AI Studio -> https://aistudio.google.com/apikey (or Groq / OpenRouter / GitHub Models, see `.env.example`).
   On Windows you can just double-click `run_all.bat`: it creates `.env`, opens it for your key, checks it, and runs everything.
3. Copy `.env.example` to `.env` and paste your key:
   - Windows: `copy .env.example .env` then open `.env` in Notepad
   - Mac/Linux: `cp .env.example .env`
4. Put the episode files in `./data/` (unzip the input pack there; sub-folders are fine).
5. Check the key and model work (1 request):
   ```
   python check_setup.py
   ```

## Run

```
python run.py --input ./data --out test_out/ --episodes 1-4     # quick test, 3 requests
python run.py --input ./data --out outputs/                     # full show, ~30 requests, ~10-20 min
```

* Every LLM answer is saved in `outputs/raw_llm/`. If the run stops (for example the free daily quota
  runs out), run the same command again: saved answers are reused and it continues where it stopped.
  You can also switch to another model/provider in `.env` and continue. `--fresh` ignores saved answers.
* All settings (window size, weights, penalties, lengths, prices) are in `config.py`.
* Free-tier models are sometimes overloaded. `python speed_test.py gemini-3.6-flash gemini-flash-latest ...`
  times one tiny request per model so you can pick a responsive one.
* **Swapping the model:** change the lines in `.env`. `LLM_PROVIDER=gemini` uses Google's native
  Gemini API; `LLM_PROVIDER=openai` works with any OpenAI-compatible API (Groq, OpenRouter, GitHub
  Models, OpenAI, a local Ollama...). Only `llm.py` talks to the model, so nothing else changes.

## Outputs (`outputs/`, written by the script, never edited by hand)

| File | What it is |
|---|---|
| `ranked_promos.md` | The ranked list to read: segments (mm:ss), hook, what plays in the first 3 s, cliffhanger, last line, why it works, why post it at that rank, scores, flags |
| `promos.json`, `promos.csv` | The same, machine-readable |
| `rejected.csv` | Every candidate that was dropped, and why |
| `funnel.csv` | Counts: proposed -> valid -> unique -> judged -> ranked, plus rejection reasons |
| `cost.csv`, `run_log.txt` | Calls, tokens, estimated cost, runtime, scale-up estimate, top 10 |
| `candidates_all.csv`, `candidates_valid.csv` | Every candidate before / after the checks |
| `episodes.csv`, `story_summary.json` | Parsed episode stats; the LLM's story summary |
| `raw_llm/` | Every unedited LLM answer (the prompts are not saved because they contain the script) |
| `labels.csv`, `my_picks.csv` | Templates for the human check (below) |

## Quality check (human in the loop)

1. Open `outputs/labels.csv` in **Notepad or VS Code** (Excel changes `00:40.00` into a time and breaks it).
   For each of the top 20 promos write `good`, `ok` or `bad` in `my_rating` and a short `my_note`.
2. Open `outputs/my_picks.csv`. For each listed episode, read the script and write the moment you
   would pick yourself (`start`, `end` as mm:ss). Add rows for more picks.
3. `python eval.py --out outputs` -> precision@10/@20, recall of my picks, Spearman(score, my rating),
   saved to `outputs/eval_results.md`.
4. `python consistency.py --input ./data --out outputs --episodes 1-7` -> asks the same FIND question
   again (2 requests) and reports how many promos are found both times (`outputs/consistency.md`).
5. `python tests/smoke_test.py --input ./data --episodes 1-8` -> tests the plumbing with a fake LLM
   (no API, no cost); it checks that invented times, too-short clips, duplicates and mid-sentence cuts are caught.

## How this scales to every show

* Calls grow linearly with episodes: about 1 FIND call per 3 episodes + 1 JUDGE call per 8 clips +
  2 calls per show, i.e. roughly **0.6 calls per episode**. Measured tokens and cost are in
  `outputs/cost.csv`, including an estimate for 100 shows x 50 episodes.
* With a paid key, windows and judge batches are independent, so they can run in parallel; a show
  takes about a minute instead of 5-10 (the free tier forces a wait between calls).
* The tool only needs the screenplay format Kuku already produces (timed lines). A new show = a new folder.
* A human editor stays in the loop: the tool shortlists and explains; the editor watches the top
  clips and approves. Their good/bad decisions are the labels to tune the prompts and weights later.

## Results of the full run (50 episodes)

| | |
|---|---|
| Candidates proposed / passed checks / unique / ranked | 101 / 101 / 81 / 81 |
| API calls, tokens (in / out) | 26 calls, 186k / 142k |
| Cost at paid prices ($0.30 / $2.50 per M tokens) | $0.41 for the show, $0.008 per episode, ~$41 for 100 shows x 50 episodes |
| Independent review of the top 20 (a second model) | precision@10 60% good (90% good or ok); precision@20 50% (90%) |
| Recall of the reviewer's own picks (eps 1, 10, 20, 30, 40, 50) | 4 of 6 found, 2 in the top 20 |
| Spearman(tool score, review rating), top 20 | 0.06 - a good shortlist, but the order inside it is close to random |
| Consistency (same FIND question asked twice, eps 40-46) | 75% of promos found again |

**Known failures** (details with promo ids in `FAILURES.md`, shared with the outputs):
the best entry-point clip (episode 1, jilted at the altar) ranks only #20; near-duplicate clips survive
(26% overlap is under the 50% threshold); slow openings rank high when the cliffhanger is strong; one spoiler was
missed; the "makes sense alone" score is 4/5 for 70 of 81 clips, so it barely separates them.

`FAILURES.md`, the solution PDF and the run outputs quote Kuku's confidential script, so they are shared
privately with the submission instead of being published here.

## Limits

* It reads text only: it cannot see acting, faces, music or how a slap actually looks.
* Scores come from an LLM, so they vary between runs (see `consistency.md`) and can be generous.
* Rules like "no spoilers" depend on the LLM's story summary being right.

## Confidentiality

The screenplays (`data/`) and the run outputs (which quote the script) are excluded by `.gitignore`
and are not in this public repo. The outputs are shared privately via Google Drive.
