"""
All the knobs in one place. Change these, not the code.
(The model / API key are NOT here - they come from the .env file so the model is easy to swap.)
"""

# ---------- STEP 2: FIND ----------
# The LLM reads a "window" of consecutive episodes at a time.
# Windows overlap by 1 episode, so every pair of neighbouring episodes (7+8, 8+9, ...)
# is seen together at least once -> cliffhangers that cross an episode boundary are visible.
# 4 episodes per window = 17 calls for 50 episodes (pairs would need 49 calls, too many
# for a free tier that allows ~20-50 requests per day).
EPISODES_PER_WINDOW = 4
MAX_PROMOS_PER_WINDOW = 6
INCLUDE_CAMERA_LINES = False  # camera directions double the tokens and add little

# ---------- STEP 3: CHECK ----------
MIN_PROMO_SECONDS = 20.0        # from the brief: promos are 20-60 seconds
MAX_PROMO_SECONDS = 60.0
MAX_SEGMENTS = 3                # a promo may join at most 3 moments
TIMESTAMP_TOLERANCE = 1.0       # LLM time must be within 1 s of a real line time, else "invented"
DUPLICATE_OVERLAP = 0.5         # two promos sharing >= 50% of their time are duplicates
HOOK_WINDOW_SECONDS = 3.0       # the hook must happen in the first 3 seconds

# ---------- STEP 4: JUDGE ----------
CLIPS_PER_JUDGE_CALL = 12       # judge several clips per call to save requests (free tier: ~20 requests/day per model)

# ---------- STEP 5: RANK ----------
# final score = weighted sum of the three 1-5 scores, minus penalties.
# Equal weights: the brief lists the three qualities without saying one matters more.
WEIGHTS = {"hook_strength": 1.0, "cliffhanger_strength": 1.0, "makes_sense_alone": 1.0}
PENALTY_NEEDS_SETUP = 2.0       # cold viewer could not follow the clip on its own
PENALTY_SPOILER = {"low": 0.0, "medium": 1.0, "high": 2.0}  # gives away a big late twist
TOP_N = 10
MAX_SAME_TYPE_IN_TOP = 3        # diversity: at most 3 "slap" (or any one type) clips in the top 10
SAME_MINUTE_SECONDS = 60.0      # diversity: no two top clips starting within 60 s in the same episode

# ---------- LLM call behaviour ----------
TEMPERATURE = 0.4
SECONDS_BETWEEN_CALLS = 7       # free tiers allow ~5-10 requests per minute
RETRY_WAITS = [15, 30, 60, 90]  # seconds to wait after a rate-limit / server error
JSON_MODE = True                # ask the API for JSON output (turned off automatically if unsupported)

# ---------- Cost estimate (edit to the current price of the model you use) ----------
# Assumed paid list price for a Flash-class model, USD per 1 million tokens.
PRICE_IN_PER_M = 0.30
PRICE_OUT_PER_M = 2.50
SCALE_SHOWS = 100               # "every show we make": assume 100 shows x 50 episodes
SCALE_EPISODES_PER_SHOW = 50
