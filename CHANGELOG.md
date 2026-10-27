# Changelog

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
