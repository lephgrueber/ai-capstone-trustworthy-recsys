# What the first OPE experiment tells us

This analysis uses the saved model and completed evaluation. It changes no model, policy or test split.

## Why the result is uncertain

| Partition | Recorded recommendations | All clicks | Records matching the model's favorite item | Clicks on that favorite |
|---|---:|---:|---:|---:|
| train | 366,404 | 1288 | 4737 | 121 |
| validation | 45,800 | 173 | 585 | 2 |
| test | 45,801 | 161 | 572 | 4 |

For the strongest personalized candidate, 4 clicks account for 94.84% of the weighted click total. The old system picked randomly among 80 items, so it rarely displayed the particular item this candidate favored. Each such observation receives substantial influence when estimating the alternative approach.

One of those clicks represents 23.71% of the weighted total. Changing that one click to a non-click would move its IPS estimate to 0.005065. This is a sensitivity illustration, not a reason to remove a legitimate click.

## Which items were favored

IDs below belong to the Open Bandit fashion catalog, not MovieLens. Counts refer to the saved model's top-scoring action for each context before exploration, not actual recommendations delivered by our application.

| Action ID | Test contexts preferring it | Times it was actually shown in those contexts | Clicks in those matches |
|---|---:|---:|---:|
| 43 | 15892 | 198 | 1 |
| 35 | 8031 | 97 | 2 |
| 62 | 5516 | 62 | 0 |
| 7 | 5346 | 71 | 0 |
| 60 | 4263 | 56 | 1 |
| 41 | 2310 | 25 | 0 |
| 47 | 2165 | 31 | 0 |
| 24 | 691 | 11 | 0 |
| 26 | 466 | 7 | 0 |
| 38 | 455 | 6 | 0 |

## What to conclude

Validation chose uniform selection. Keep that recorded decision. Higher personalized point estimates on the final test do not justify switching winners after inspecting it. The simple model may be inaccurate, preferences may vary over time, and there are few informative clicks; these data do not isolate which explanation caused the result.

The narrow intervals from the prediction-only method do not mean the model is accurate: they hold its predictions fixed and omit model bias and fitting uncertainty. The importance-weighted estimates and wide intervals are a useful warning.

## Next experimental steps

Preserve this experiment as the initial benchmark. Any revised model should be developed on training/validation data and assessed with a predeclared fresh evaluation window or replication campaign. Label further analysis of this test as exploratory. For the movie app, implement the separate logging design in movie_feedback_plan.md before claiming movie-policy OPE. Do not apply Open Bandit item IDs, click rates, or the selected uniform policy to the movie application.
