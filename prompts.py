"""
All the text we send to the LLM, in one file, so the prompts are easy to read and tune.
"""

# ---------------------------------------------------------------- SUMMARY (1 call)
SUMMARY_SYSTEM = "You are a story editor for short vertical drama series. You answer in JSON only."

SUMMARY_USER = """Below are the one-line loglines of every episode of the series "{title}".

{loglines}

Return JSON:
{{
  "summary": "150-200 words: the premise and main story arc",
  "characters": [{{"name": "...", "who": "one line: who they are and their relationship to the lead"}}],
  "big_reveals": [{{"episode": 12, "reveal": "a major twist or secret revealed in that episode"}}]
}}
List the main characters (max 10) and the 5-8 biggest twists/reveals of the whole series."""

# ---------------------------------------------------------------- FIND (1 call per window)
FIND_SYSTEM = (
    "You are a promo editor for vertical microdrama apps like ReelShort and DramaBox. "
    "You choose 20-60 second clips from a show's own footage to post as ads on TikTok, Instagram Reels "
    "and YouTube Shorts, to make scrolling strangers stop and start watching. You answer in JSON only."
)

FIND_USER = """SHOW: "{title}"
STORY SUMMARY: {summary}
MAIN CHARACTERS: {characters}
BIG REVEALS (do not give these away cheaply): {reveals}

WHAT A GOOD PROMO IS
1. HOOK: something in the FIRST 3 SECONDS that stops a scrolling viewer - a slap, a shocking line, a reveal, a threat, a humiliation.
2. CLIFFHANGER: it ENDS on an unresolved moment - cut right BEFORE the answer, the reaction or the outcome.
3. STANDS ALONE: a person who has never seen the show understands who is fighting whom and what is at stake.

RULES FOR TIMESTAMPS (very important)
- Every script line below starts with [E<episode> start-end]. Use ONLY those times. Never invent a time.
- A segment's "start" must be the START time of a line; its "end" must be the END time of a line.
- Each segment lies inside one episode. A promo has 1 to {max_segments} segments in story order.
  Use more than one segment only when it really helps (e.g. the cliffhanger at the end of one episode
  + the shocking opening of the next, or a hook line joined to the payoff a little later).
- The TOTAL length of all segments must be between {min_s:.0f} and {max_s:.0f} seconds (aim for 25-45).
  Before answering, add up (end - start) of every segment: a 2-3 second segment adds very little, so most promos
  should be one or two long continuous stretches.
- Look especially at episode endings: they are written as cliffhangers.

Find up to {max_promos} DIFFERENT promos in the episodes below (different moments, not small variations).

EPISODES IN THIS WINDOW
{episodes_text}

Return JSON:
{{
  "promos": [
    {{
      "segments": [{{"episode": 7, "start": "01:12.40", "end": "01:48.00"}}],
      "hook": "what happens in the first 3 seconds",
      "cliffhanger": "the unresolved moment it cuts on",
      "why_it_works": "one line",
      "moment_type": "slap | betrayal | reveal | humiliation | threat | romance | rescue | confrontation | other",
      "spoiler_risk": "low | medium | high (high = the clip itself gives away one of the BIG REVEALS)"
    }}
  ]
}}"""

# ---------------------------------------------------------------- JUDGE (batched)
JUDGE_SYSTEM = (
    "You are two people at once: (1) a stranger scrolling TikTok who has NEVER seen this show and knows "
    "nothing about it, and (2) a strict, honest promo reviewer. Most clips are average; do not give 5s easily. "
    "You answer in JSON only."
)

JUDGE_USER = """Below are {n} short clips cut from a vertical drama. Each clip shows only its own lines
(dialogue and action) with times from the start of the clip. Lines marked >>> happen in the first 3 seconds.
"--- CUT ---" means the clip jumps to a different moment. You know NOTHING else about the story.

{clips}

For EACH clip, first read it as the stranger, then score it as the reviewer:
- who_are_they: your best guess, from this clip only
- what_is_happening: one line
- what_is_unresolved: the open question the clip leaves you with ("nothing" if it resolves)
- understands_alone: true if a first-time viewer could follow who is against whom and what is at stake
- hook_strength 1-5, judging ONLY the >>> lines (first 3 s): 5 = slap / shocking line / big reveal instantly;
  3 = intriguing but slow; 1 = quiet set-up, scenery, or nothing happens
- cliffhanger_strength 1-5, judging the LAST line/moment: 5 = cuts right before the answer to a big question or threat;
  3 = some tension; 1 = the moment resolves or ends calmly
- makes_sense_alone 1-5: 5 = anyone gets who, what and the stakes; 3 = gets the gist, some confusion; 1 = needs earlier episodes
- give ONE sentence of evidence for each score

Return JSON:
{{
  "clips": [
    {{
      "clip_id": "P001",
      "who_are_they": "...", "what_is_happening": "...", "what_is_unresolved": "...",
      "understands_alone": true,
      "hook_strength": 4, "hook_evidence": "...",
      "cliffhanger_strength": 3, "cliffhanger_evidence": "...",
      "makes_sense_alone": 4, "sense_evidence": "..."
    }}
  ]
}}"""

# ---------------------------------------------------------------- WHY POST FIRST (1 call)
WHY_SYSTEM = "You are the social media lead for a microdrama app. You answer in JSON only."

WHY_USER = """These are the top {n} promos for the show "{title}", in the order we plan to post them
(rank 1 = post first). Scores are 1-5.

{promos}

For each promo write ONE line (max 25 words) on why it deserves its place in the posting order -
what it gives a brand-new viewer and why it would make them start the show.

Return JSON: {{"why": [{{"promo_id": "P001", "why_post_first": "..."}}]}}"""
