# Changelog

## Unreleased

### Token and call accounting

Fixes the three accounting issues listed as known issues under v1.1. Runs made with earlier
versions, including the paper's, were counted with the old behaviour.

- Claude native backend (`claude_ptc`): in multi-turn tasks, a code-execution turn that ended
  while a follow-up user message was waiting was recorded twice, adding its tokens and an
  extra assistant turn a second time. It is now recorded once.
- Gemini backends (`gemini`, `gemini_native`): output tokens now include thinking tokens
  (`thoughts_token_count`), which Gemini reports separately but bills as output. This matches
  the OpenAI and Anthropic backends, whose output counts already include reasoning tokens.
  Gemini raw-token counts are therefore higher than before, and a `--max-token-budget`, if
  set, is reached sooner.
- Trajectory tokens now count every user message once, including the follow-up question in
  multi-turn tasks; previously only the first message was counted.
- Added unit tests for these cases (`tests/`, standard-library `unittest`, no API keys).

## v1.2

### Benchmark data (main split)

An audit of the main-split tasks that no model or interface had ever passed found nine more
whose golden answers or verifiers made a correct answer fail. As with v1.1, every interface
failed them equally, so the paper's comparisons are unaffected; its absolute main-split pass
rates are understated. Prompts are unchanged.

- `ptc_code_0`: the verifier hard-coded figures from an older, smaller version of the scenario
  (alice_v, $282, 43.6%). It now computes them from the data. The nine Venmo transactions dated
  after the scenario's "today" (2026-03-31) were removed, which made "the past month"
  ambiguous; the calendar month and the trailing 30 days are both accepted. The message to the
  top recipient may be a payment or a payment request.
- `ptc_code_2`: the verifier hard-coded an older total ($443). It now computes the past three
  months from the scenario's date ($588.50 over 25 rides, most expensive $44.00), accepting the
  three natural readings of "past 3 months", and only expects the budget event and Venmo
  request if the monthly average exceeds $200.
- `ptc_combined_3`: the verifier hard-coded older totals ($450 / $160 / $111.38); it now computes
  them from the data (Venmo $2859.50, Uber $1421.90, or $1611.00 counting cancelled rides,
  DoorDash $1352.50). The self-payment the prompt asks for received id `txn_1`, which already
  existed, so it could never be detected; the scenario now sets `transaction_counter` so new
  payments get fresh ids, and the verifier detects new payments by position.
- `ptc_org_chart_most_teammates_0`: the prompt counts teammates only for overlapping periods
  under the same manager. That gives 23 for Alice; the golden answer said 30, which counts
  anyone who ever shared a manager with her.
- `ptc_yelp_uber_pareto_chinese_0`: the prompt defines strict dominance (higher rating and
  lower Uber cost). 17 of the 21 restaurants tie at the $8.00 minimum fare, so 17 survive and
  4 are dominated; the golden answer used the looser Pareto rule (1 survivor, 20 dominated). The
  recommended restaurant is unchanged.
- `ptc_queries_shared_supervisor_pairs_0`: 21 pairs shared two supervisors, and the verifier kept
  only the last entry per pair, so the result depended on output order. Entries are now matched
  by pair and supervisor; a pair may be listed once per supervisor or with any one of them.
- `ptc_yelp_category_best_nearby_0`: the verifier compared category names case-sensitively
  against lowercase keys, so an answer using the API's spelling failed. It now recomputes the
  answer from the data, compares names case-insensitively, accepts any business tied for a
  category's top rating, and treats the Bakery, Cafe, Dessert and Bar categories (whose
  businesses are not tagged "restaurant") as optional.
- `ptc_queries_calendar_five_attendees_0`: no 45-minute slot fit all five attendees' 10 a.m. to
  3 p.m. local windows (Los Angeles and Berlin never overlap), and the verifier also checked
  seven employees the prompt does not mention. Diana now works in America/Denver and Eric in
  America/New_York, giving a common window of 17:00 to 19:00 UTC. The verifier checks only the
  five attendees, accepts timestamps with or without seconds, and now enforces "fewest
  conflicts": a weekday slot must match the weekday minimum, and a weekend slot (next week's
  weekends have no events) must match the overall minimum.
- `ptc_queries_calendar_interview_scheduling_0`: the starting calendars already contain 71
  overlapping event pairs that the API has no way to remove, so the no-overlap check could
  never pass. It now counts only overlaps involving a newly booked interview. Interviews that
  share a start time and title are now split into candidate-interviewer pairs instead of being
  merged into one booking.

## v1.1

### Benchmark data (main split)

Six tasks had golden answers or verifiers that contradicted their own prompts, so an agent
that followed the instructions correctly could not pass them. In our own development runs
they were essentially never passed by any model or interface. The paper's results were
computed with v1.0, so its absolute main-split pass rates are slightly understated, equally
for every interface; comparisons between interfaces are unaffected.

- `ptc_yelp_top_rated_reviews_0`: the prompt asks for restaurants in Portland, but every
  business had a San Francisco address, so a location search returned nothing. The ten
  hand-built restaurants now have Portland addresses; the 75 "Portland Eatery" businesses stay
  in San Francisco as distractors. The golden answer is now Le Canard, Nob Hill Sushi and Pine
  Street Bistro with their two newest reviews. It previously had no reviews at all, so the
  verifier crashed on every answer. The verifier now matches the two reviews in either order.
- `ptc_yelp_chinese_near_boba_0`: golden pairs regenerated with the prompt's 0.3-mile walking
  limit. There are 4 qualifying pairs; 10 of the previous 14 were 0.48 to 0.88 miles apart.
- `ptc_yelp_min_stops_coverage_0`: the golden answer used three stops, but a unique one-stop
  solution exists (Thai Orchid, Bubble Pop, Ice Cream Palace; combined rating 13.2), and the
  prompt minimizes stops first. The verifier now also accepts the category label "dinner" that
  the prompt itself uses; it previously rejected the golden answer's own labels.
- `ptc_yelp_cascading_search_0`: "a wait time under 20 minutes" now excludes 20. The answer is
  Restaurant 51 (`biz_51`, 10-minute wait) instead of Restaurant 39 (20-minute wait).
- `ptc_yelp_highest_rated_filtered_0`: the golden answer is now Lucky Wok (`biz_10`; 4.9 stars,
  200 reviews, 9.76 miles, open), the highest-rated restaurant meeting the prompt's criteria.
  It was previously Wok This Way (4.4 stars).
- `ptc_yelp_cascading_search_0` and `ptc_yelp_highest_rated_filtered_0`: saving the restaurant
  with either `save_business()` or `bookmark_business()` now counts. The prompts say
  "bookmarks", but previously only `save_business()` was accepted.

### Harness

- `run_eval`: in the default in-process mode, a task that finishes after the run-level
  timeout no longer overwrites its recorded timeout trace, so the trace files and the summary
  agree.
- `run_eval`: the native backends (`claude_ptc`, `openai_native`, `gemini_native`) now reject
  `--tool-inspection` and `--mode iptc`. Inspection hides the per-method API tools, which left
  native runs with no API tools at all.
- `requirements.txt`: minimum SDK versions raised to the versions the harness was developed
  and run with.
- README: added a security note. Model-generated code runs in-process with full builtins; it
  is not sandboxed.

### Known issues, not changed in this release

- The Gemini backends count `candidates_token_count` as output tokens, which excludes thinking
  tokens, so Gemini raw-token counts are understated.
- The Claude native backend (`claude_ptc`) counts the last response of the first turn twice in
  multi-turn tasks.
- Trajectory tokens count only the first user message of multi-turn tasks.

## v1.0

Initial release; the code and data used for the paper.
