# Way4 M0 - Regression Baselines

_Generated (UTC): 2026-09-11T11:15:55+00:00_

Fixed-seed regression benchmark for the pre-Way4 stacks (DESIGN.md section 14, M0). Way4 changes must not regress these numbers.

- **way2** = `radio_rl.pipeline.Pipeline` -- all five metrics incl. channel switch.
- **way3** = `jammerhunt` runner policy on the `offline_sim` engine bridge (`field_kind='smooth'`, `mode='formal'`).
- Seeds: P3 = `range(1000,1010)` (10 seeds), P4 = `range(2000,2008)` (8 seeds); same ranges for both stacks.
- Metrics: success, time (virtual s), move (m), measure (# `/measure`), switch (# channel switches), cleared/total.
- clear-success rate = fraction of episodes with `success=True` (all sources cleared).
- Percentiles use linear interpolation (same method as `offline_sim.harness._pct`).
- **Gap**: channel switches are **not tracked** in the Way3/offline_sim bridge (`EpisodeResult` has no switch field) -> recorded as `null`, shown as —.
- estimator suite skipped (source not on this branch).

## Aggregate summary

| Stack | Prob | Eps | Clear-success | Cleared/Total | Time P50 | P90 | P95 | Max | Mean move | Mean measure | Mean switch |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| way2 | P3 | 10 | 80.0% (8/10) | 136/138 | 9883.4 | 10350.7 | 10388.3 | 10425.8 | 45044.7 | 102.30 | 86.50 |
| way2 | P4 | 8 | 0.0% (0/8) | 0/104 | 360000.0 | 360000.9 | 360002.0 | 360003.0 | 0.0 | 72001.00 | 0.38 |
| way3 | P3 | 10 | 100.0% (10/10) | 138/138 | 5122.9 | 5787.6 | 6119.2 | 6450.9 | 21364.8 | 139.10 | — |
| way3 | P4 | 8 | 100.0% (8/8) | 104/104 | 9994.1 | 10616.9 | 10726.4 | 10835.9 | 37288.1 | 418.00 | — |

## way2 - P3

Seeds `1000..1009` (10 ok / 0 error of 10). Time mean 9674.9 s.

| Seed | Success | Cleared/Total | Time (s) | Move (m) | Measure | Switch | Note |
|---|---|---|---:|---:|---:|---:|---|
| 1000 | yes | 16/16 | 10425.8 | 48679.1 | 105 | 85 |  |
| 1001 | yes | 16/16 | 9989.0 | 46885.2 | 92 | 72 |  |
| 1002 | yes | 14/14 | 8294.2 | 38485.8 | 91 | 72 |  |
| 1003 | yes | 13/13 | 10342.4 | 48426.8 | 101 | 87 |  |
| 1004 | yes | 13/13 | 9907.9 | 46149.7 | 105 | 88 |  |
| 1005 | yes | 13/13 | 9042.2 | 41736.0 | 107 | 95 |  |
| 1006 | yes | 12/12 | 10122.8 | 47063.8 | 110 | 100 |  |
| 1007 | **NO** | 15/16 | 9858.9 | 45664.5 | 112 | 91 |  |
| 1008 | yes | 14/14 | 8998.5 | 41597.7 | 104 | 89 |  |
| 1009 | **NO** | 10/11 | 9767.7 | 45758.4 | 96 | 86 |  |

## way2 - P4

Seeds `2000..2007` (8 ok / 0 error of 8). Time mean 360000.4 s.

| Seed | Success | Cleared/Total | Time (s) | Move (m) | Measure | Switch | Note |
|---|---|---|---:|---:|---:|---:|---|
| 2000 | **NO** | 0/13 | 360000.0 | 0.0 | 72001 | 0 |  |
| 2001 | **NO** | 0/14 | 360000.0 | 0.0 | 72001 | 0 |  |
| 2002 | **NO** | 0/15 | 360000.0 | 0.0 | 72001 | 0 |  |
| 2003 | **NO** | 0/10 | 360000.0 | 0.0 | 72001 | 0 |  |
| 2004 | **NO** | 0/14 | 360000.0 | 0.0 | 72001 | 0 |  |
| 2005 | **NO** | 0/13 | 360003.0 | 0.0 | 72001 | 3 |  |
| 2006 | **NO** | 0/14 | 360000.0 | 0.0 | 72001 | 0 |  |
| 2007 | **NO** | 0/11 | 360000.0 | 0.0 | 72001 | 0 |  |

## way3 - P3

Seeds `1000..1009` (10 ok / 0 error of 10). Time mean 5167.6 s.

| Seed | Success | Cleared/Total | Time (s) | Move (m) | Measure | Switch | Note |
|---|---|---|---:|---:|---:|---:|---|
| 1000 | yes | 16/16 | 6450.9 | 27284.4 | 155 | — |  |
| 1001 | yes | 16/16 | 5713.9 | 24109.4 | 137 | — |  |
| 1002 | yes | 14/14 | 4434.2 | 17991.1 | 129 | — |  |
| 1003 | yes | 13/13 | 4893.6 | 19992.9 | 140 | — |  |
| 1004 | yes | 13/13 | 4524.7 | 18468.3 | 129 | — |  |
| 1005 | yes | 13/13 | 5118.9 | 21109.6 | 140 | — |  |
| 1006 | yes | 12/12 | 5636.9 | 23379.5 | 152 | — |  |
| 1007 | yes | 16/16 | 5126.9 | 20959.4 | 144 | — |  |
| 1008 | yes | 14/14 | 5211.6 | 21757.8 | 133 | — |  |
| 1009 | yes | 11/11 | 4564.1 | 18595.4 | 132 | — |  |

## way3 - P4

Seeds `2000..2007` (8 ok / 0 error of 8). Time mean 10022.9 s.

| Seed | Success | Cleared/Total | Time (s) | Move (m) | Measure | Switch | Note |
|---|---|---|---:|---:|---:|---:|---|
| 2000 | yes | 13/13 | 10084.2 | 37681.0 | 415 | — |  |
| 2001 | yes | 14/14 | 9431.1 | 36300.6 | 352 | — |  |
| 2002 | yes | 15/15 | 9395.5 | 36527.7 | 337 | — |  |
| 2003 | yes | 10/10 | 10835.9 | 38434.4 | 518 | — |  |
| 2004 | yes | 14/14 | 9538.6 | 35552.9 | 394 | — |  |
| 2005 | yes | 13/13 | 10523.1 | 39670.5 | 422 | — |  |
| 2006 | yes | 14/14 | 10470.5 | 38427.6 | 454 | — |  |
| 2007 | yes | 11/11 | 9903.9 | 35709.7 | 452 | — |  |

