#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Way4 milestone M0 -- regression baseline recorder.

Records per-episode metrics for the two pre-Way4 stacks on fixed seeds and
writes machine-readable JSON plus a human-readable BASELINE.md, all under Way4\\.
This is the frozen regression benchmark referenced by DESIGN.md section 14 (M0):
Way4 changes must not regress these numbers.

Stacks
------
  way2 = radio_rl.pipeline.Pipeline
         gives all five metrics including channel switches.
  way3 = jammerhunt runner policy driven on the offline_sim engine bridge
         (field_kind='smooth', mode='formal'); gives success/time/move/measure
         and cleared/total, but channel switches are NOT tracked -> recorded null.

Per-episode metrics: success(bool), time(virtual seconds), move(distance m),
measure(# of /measure calls), switch(# channel switches), cleared, total.

Run (script fixes sys.path, so cwd does not strictly matter, but the intended
invocation is from D:\\George\\SX):
    python Way4\\scripts\\record_baseline.py

Outputs (only under D:\\George\\SX\\Way4\\):
    baselines\\way2_p3.json  way2_p4.json  way3_p3.json  way3_p4.json
    BASELINE.md

estimator suite is intentionally skipped (source not on this branch).
No files outside Way4\\ are written; no git operations are performed.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import statistics
import sys
import traceback
from pathlib import Path

# --- import roots --------------------------------------------------------
# Running a script *by path* puts the script's own dir (Way4\scripts) on
# sys.path[0], NOT the cwd -- so add the roots explicitly. Way3 has no
# offline_sim package of its own, so there is no shadowing; SX_ROOT is put
# on top anyway so the canonical offline_sim wins.
SX_ROOT = r"D:\George\SX"
WAY3_ROOT = r"D:\George\SX\Way3"
for _p in (WAY3_ROOT, SX_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:  # make unicode (em dash etc.) safe on a legacy Windows console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

WAY4 = Path(r"D:\George\SX\Way4")
BASELINES_DIR = WAY4 / "baselines"
BASELINE_MD = WAY4 / "BASELINE.md"

# Way3 seeds are authoritative; Way2 reuses the same ranges for comparability.
SEED_RANGES = {
    ("way2", 3): range(1000, 1010),
    ("way2", 4): range(2000, 2008),
    ("way3", 3): range(1000, 1010),
    ("way3", 4): range(2000, 2008),
}
COMBOS = [("way2", 3), ("way2", 4), ("way3", 3), ("way3", 4)]
SWITCH_TRACKED = {"way2": True, "way3": False}


# --- metric helpers ------------------------------------------------------
def _pct(sorted_vals, q):
    """Linear-interpolation percentile (matches offline_sim.harness._pct)."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = q * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# --- per-episode runners -------------------------------------------------
def _run_way2(problem, seed):
    from radio_rl.core.config import compose
    from radio_rl.pipeline import Pipeline
    cfg = compose(overrides=[f"problem={problem}", f"seed={seed}"])
    stats, _ = Pipeline(cfg).run_episode(seed=seed)
    return {
        "seed": seed,
        "success": bool(stats.success),
        "time": float(stats.virtual_time),
        "move": float(stats.route_distance),
        "measure": int(stats.num_scans),
        "switch": int(stats.num_switches),
        "cleared": int(stats.sources_cleared),
        "total": int(stats.sources_total),
        "error": None,
    }


def _run_way3(problem, seed):
    from offline_sim.case import generate_case
    from offline_sim.harness import run_episode
    from jammerhunt.agent import make_runner_policy
    c = generate_case(seed=seed, problem=problem, field_kind="smooth", mode="formal")
    r = run_episode(c, make_runner_policy(problem))
    return {
        "seed": seed,
        "success": bool(r.success),
        "time": float(r.virtual_time_s),
        "move": float(r.move_distance_m),
        "measure": int(r.n_measure),
        "switch": None,               # not tracked in the Way3/offline_sim bridge
        "cleared": int(r.cleared),
        "total": int(r.total),
        "n_clear": int(r.n_clear),
        "n_clear_fail": int(r.n_clear_fail),
        "policy_error": r.error,      # harness records a policy exception here
        "error": None,
    }


RUNNERS = {"way2": _run_way2, "way3": _run_way3}


def record_batch(stack, problem, seeds):
    runner = RUNNERS[stack]
    rows = []
    for seed in seeds:
        try:
            row = runner(problem, seed)                 # noqa: PERF203
        except Exception as e:  # one bad seed must not abort the whole batch
            traceback.print_exc()
            row = {
                "seed": seed, "success": None, "time": None, "move": None,
                "measure": None, "switch": None, "cleared": None, "total": None,
                "error": f"{type(e).__name__}: {e}",
            }
        tag = "OK " if row.get("error") is None else "ERR"
        pol = row.get("policy_error")
        extra = f" policy_error={pol}" if pol else ""
        print("  [{} p{}] seed={} {} success={} time={}{}".format(
            stack, problem, seed, tag, row.get("success"), row.get("time"), extra))
        rows.append(row)
    return rows


def aggregate(rows, stack):
    ok = [r for r in rows if r.get("error") is None and r.get("success") is not None]
    n = len(rows)
    n_ok = len(ok)
    n_success = sum(1 for r in ok if r["success"])
    times = sorted(float(r["time"]) for r in ok)
    switches = [r["switch"] for r in ok if r.get("switch") is not None]
    return {
        "n_episodes": n,
        "n_evaluated": n_ok,
        "n_errors": n - n_ok,
        "n_success": n_success,
        "clear_success_rate": (n_success / n_ok) if n_ok else None,
        "sources_cleared": sum(int(r["cleared"]) for r in ok) if n_ok else 0,
        "sources_total": sum(int(r["total"]) for r in ok) if n_ok else 0,
        "time_mean": statistics.fmean(times) if times else None,
        "time_p50": _pct(times, 0.50),
        "time_p90": _pct(times, 0.90),
        "time_p95": _pct(times, 0.95),
        "time_max": times[-1] if times else None,
        "move_mean": statistics.fmean(float(r["move"]) for r in ok) if n_ok else None,
        "measure_mean": statistics.fmean(int(r["measure"]) for r in ok) if n_ok else None,
        "switch_mean": statistics.fmean(switches) if switches else None,
        "switch_tracked": SWITCH_TRACKED[stack],
    }


# --- formatting ----------------------------------------------------------
def _f(v, nd=1):        # markdown formatter (em dash for missing)
    return "\u2014" if v is None else f"{v:.{nd}f}"


def _fa(v, nd=1):       # ascii formatter for console
    return "n/a" if v is None else f"{v:.{nd}f}"


def _pctstr(v):
    return "\u2014" if v is None else f"{v * 100:.1f}%"


def write_markdown(data, now):
    L = []
    L.append("# Way4 M0 - Regression Baselines\n")
    L.append(f"_Generated (UTC): {now}_\n")
    L.append("Fixed-seed regression benchmark for the pre-Way4 stacks (DESIGN.md "
             "section 14, M0). Way4 changes must not regress these numbers.\n")
    L.append("- **way2** = `radio_rl.pipeline.Pipeline` -- all five metrics incl. "
             "channel switch.")
    L.append("- **way3** = `jammerhunt` runner policy on the `offline_sim` engine "
             "bridge (`field_kind='smooth'`, `mode='formal'`).")
    L.append("- Seeds: P3 = `range(1000,1010)` (10 seeds), P4 = `range(2000,2008)` "
             "(8 seeds); same ranges for both stacks.")
    L.append("- Metrics: success, time (virtual s), move (m), measure (# `/measure`), "
             "switch (# channel switches), cleared/total.")
    L.append("- clear-success rate = fraction of episodes with `success=True` "
             "(all sources cleared).")
    L.append("- Percentiles use linear interpolation (same method as "
             "`offline_sim.harness._pct`).")
    L.append("- **Gap**: channel switches are **not tracked** in the Way3/offline_sim "
             "bridge (`EpisodeResult` has no switch field) -> recorded as `null`, "
             "shown as \u2014.")
    L.append("- estimator suite skipped (source not on this branch).\n")

    L.append("## Aggregate summary\n")
    L.append("| Stack | Prob | Eps | Clear-success | Cleared/Total | Time P50 | P90 | "
             "P95 | Max | Mean move | Mean measure | Mean switch |")
    L.append("|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for combo in COMBOS:
        a = data[combo]["aggregates"]
        L.append("| {} | P{} | {} | {} ({}/{}) | {}/{} | {} | {} | {} | {} | {} | {} | {} |".format(
            combo[0], combo[1], a["n_episodes"],
            _pctstr(a["clear_success_rate"]), a["n_success"], a["n_evaluated"],
            a["sources_cleared"], a["sources_total"],
            _f(a["time_p50"]), _f(a["time_p90"]), _f(a["time_p95"]), _f(a["time_max"]),
            _f(a["move_mean"]), _f(a["measure_mean"], 2),
            _f(a["switch_mean"], 2) if a["switch_tracked"] else "\u2014",
        ))
    L.append("")

    for combo in COMBOS:
        stack, problem = combo
        payload = data[combo]
        a = payload["aggregates"]
        seeds = payload["seeds"]
        L.append(f"## {stack} - P{problem}\n")
        L.append(f"Seeds `{seeds[0]}..{seeds[-1]}` "
                 f"({a['n_evaluated']} ok / {a['n_errors']} error of {a['n_episodes']}). "
                 f"Time mean {_f(a['time_mean'])} s.\n")
        L.append("| Seed | Success | Cleared/Total | Time (s) | Move (m) | Measure | "
                 "Switch | Note |")
        L.append("|---|---|---|---:|---:|---:|---:|---|")
        for r in payload["episodes"]:
            note = r.get("error") or r.get("policy_error") or ""
            succ = "\u2014" if r.get("success") is None else ("yes" if r["success"] else "**NO**")
            ct = "\u2014" if r.get("cleared") is None else f"{r['cleared']}/{r['total']}"
            sw = "\u2014" if r.get("switch") is None else str(r["switch"])
            meas = "\u2014" if r.get("measure") is None else str(r["measure"])
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["seed"], succ, ct, _f(r.get("time")), _f(r.get("move")),
                meas, sw, note))
        L.append("")

    BASELINE_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
    data = {}
    for stack, problem in COMBOS:
        seeds = list(SEED_RANGES[(stack, problem)])
        print(f"== recording {stack} P{problem} seeds {seeds[0]}..{seeds[-1]} ==")
        rows = record_batch(stack, problem, seeds)
        agg = aggregate(rows, stack)
        payload = {
            "stack": stack,
            "problem": problem,
            "generated_utc": now,
            "seeds": seeds,
            "switch_tracked": SWITCH_TRACKED[stack],
            "field_kind": "smooth" if stack == "way3" else None,
            "episodes": rows,
            "aggregates": agg,
        }
        out = BASELINES_DIR / f"{stack}_p{problem}.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> wrote {out}")
        data[(stack, problem)] = payload

    write_markdown(data, now)
    print(f"  -> wrote {BASELINE_MD}")

    print("\n==== AGGREGATE SUMMARY ====")
    for combo in COMBOS:
        a = data[combo]["aggregates"]
        sw = _fa(a["switch_mean"], 2) if a["switch_tracked"] else "n/a(gap)"
        print("{} P{}: clear={} ({}/{})  time P50/P90/P95/max={}/{}/{}/{}s  "
              "move={} meas={} switch={}".format(
                  combo[0], combo[1], _pctstr(a["clear_success_rate"]).replace("\u2014", "n/a"),
                  a["n_success"], a["n_evaluated"],
                  _fa(a["time_p50"]), _fa(a["time_p90"]), _fa(a["time_p95"]), _fa(a["time_max"]),
                  _fa(a["move_mean"]), _fa(a["measure_mean"], 2), sw))


if __name__ == "__main__":
    main()
