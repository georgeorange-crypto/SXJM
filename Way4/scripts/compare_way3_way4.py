"""Way3-vs-Way4 head-to-head on the authoritative Way3 engine (DESIGN.md §14 M7).

The M7 acceptance gate: **Way4 must not lower the full-clear rate** relative to the
Way3 baseline (禁止10 — never trade the full clear for speed). Speed (virtual time) is
the secondary score, compared only on episodes both stacks fully cleared.

Fair-comparison contract (both stacks see identical ground truth):

  * ONE engine implementation + ONE generator: ``jammerhunt.environment`` for both.
  * A *fresh* ``Case`` per stack per seed — the engine mutates the case in place
    (``jammer.cleared``), so each stack needs its own. ``generate_case(seed=…)`` is
    deterministic (``random.Random(seed)``), so a fresh case at the same seed has
    identical jammer channels, positions, R_eff, *and* error field.
  * Way3 runs its native agent: ``Hunter(problem).run(LocalWorld(Engine(case)))``.
  * Way4 runs the math pipeline over the same engine through the shipped adapter:
    ``Way4Pipeline(Way3EngineAdapter(Engine(case))).run()``.
  * The verdict for BOTH is ground truth ``case.cleared_count == case.total`` — never
    ``belief.all_resolved()`` (which is Way4's internal claim, not the real world).

Run:  python scripts/compare_way3_way4.py -n 50 --problem 3
      python scripts/compare_way3_way4.py -n 30 --field constant   # worst-case ±1°
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# -- path bootstrap (src-layout, no editable install; mirrors tests/conftest.py) --
_HERE = Path(__file__).resolve()
_WAY4_SRC = _HERE.parents[1] / "src"        # SX/Way4/src   (the way4 package)
_SX_ROOT = _HERE.parents[2]                 # SX/           (sxjm_core lives here)
_WAY3 = _SX_ROOT / "Way3"                    # SX/Way3       (the jammerhunt package)
for _p in (_WAY4_SRC, _SX_ROOT, _WAY3):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jammerhunt import environment as env          # noqa: E402  authoritative engine
from jammerhunt.agent import Hunter                 # noqa: E402  Way3 native strategy
from jammerhunt.interface import LocalWorld         # noqa: E402

from way4.executor import Way3EngineAdapter         # noqa: E402
from way4.pipeline import Way4Pipeline              # noqa: E402


@dataclass
class Row:
    seed: int
    stack: str                       # "way3" | "way4"
    total: int
    cleared: int
    success: bool                    # ground truth: cleared == total
    virtual_time_s: float
    wall_s: float
    steps: Optional[int] = None      # way4 only: macros executed
    finish_reason: Optional[str] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# single episodes
# --------------------------------------------------------------------------- #
def run_way3(seed: int, problem: int, field_kind: str) -> Row:
    case = env.generate_case(seed=seed, problem=problem, field_kind=field_kind)
    engine = env.Engine(case)
    err = None
    t0 = time.perf_counter()
    try:
        # Hunter.run enters the world itself (see LocalWorld.enter); mirrors mc.py.
        Hunter(problem=problem).run(LocalWorld(engine))
    except Exception as e:                    # a strategy bug must not kill the batch
        err = f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    return Row(
        seed=seed, stack="way3", total=case.total, cleared=case.cleared_count,
        success=(case.cleared_count == case.total), virtual_time_s=engine.virtual_time_s,
        wall_s=wall, finish_reason=engine.finish_reason, error=err,
    )


def run_way4(seed: int, problem: int, field_kind: str, max_steps: int) -> Row:
    # FRESH case at the same seed => identical ground truth to the Way3 run above.
    case = env.generate_case(seed=seed, problem=problem, field_kind=field_kind)
    engine = env.Engine(case)
    engine.enter()                            # the adapter requires an entered engine
    pipe = Way4Pipeline(Way3EngineAdapter(engine), n_channels=20, max_steps=max_steps)
    t0 = time.perf_counter()
    result = pipe.run()
    wall = time.perf_counter() - t0
    # Verdict from GROUND TRUTH, not belief.all_resolved().
    return Row(
        seed=seed, stack="way4", total=case.total, cleared=case.cleared_count,
        success=(case.cleared_count == case.total),
        virtual_time_s=result.virtual_time_s, wall_s=wall, steps=result.steps,
        finish_reason=engine.finish_reason, error=result.error,
    )


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _pct(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@dataclass
class StackSummary:
    stack: str
    n: int
    n_success: int
    n_error: int
    success_rate: float
    time_mean_success: float
    time_p50_success: float
    time_p90_success: float
    time_max_success: float


def summarize(rows: List[Row], stack: str) -> StackSummary:
    rows = [r for r in rows if r.stack == stack]
    n = len(rows)
    n_success = sum(1 for r in rows if r.success)
    n_error = sum(1 for r in rows if r.error)
    succ = sorted(r.virtual_time_s for r in rows if r.success)
    return StackSummary(
        stack=stack, n=n, n_success=n_success, n_error=n_error,
        success_rate=(n_success / n if n else 0.0),
        time_mean_success=(statistics.fmean(succ) if succ else 0.0),
        time_p50_success=_pct(succ, 0.50),
        time_p90_success=_pct(succ, 0.90),
        time_max_success=(succ[-1] if succ else 0.0),
    )


def _fmt_summary(s: StackSummary) -> str:
    return (
        f"  {s.stack:5s} | full-clear {s.success_rate * 100:6.2f}% "
        f"({s.n_success}/{s.n})  errors={s.n_error}  "
        f"| t_success mean/P50/P90/max = "
        f"{s.time_mean_success:7.1f}/{s.time_p50_success:7.1f}/"
        f"{s.time_p90_success:7.1f}/{s.time_max_success:7.1f} s"
    )


def compare(n: int, base_seed: int, problem: int, field_kind: str, max_steps: int,
            verbose: bool = False) -> int:
    print(f"== Way3 vs Way4  |  problem={problem}  n={n}  field={field_kind}  "
          f"seeds=[{base_seed}..{base_seed + n - 1}] ==\n")

    rows: List[Row] = []
    regressions: List[int] = []      # Way3 cleared but Way4 did not (禁止10 breach)
    for i in range(n):
        seed = base_seed + i
        r3 = run_way3(seed, problem, field_kind)
        r4 = run_way4(seed, problem, field_kind, max_steps)
        rows.extend((r3, r4))
        if r3.success and not r4.success:
            regressions.append(seed)
        if verbose or (r3.success != r4.success):
            flag = "  <-- REGRESSION" if (r3.success and not r4.success) else (
                "  (way4 wins)" if (r4.success and not r3.success) else "")
            print(f"  seed={seed:5d} tot={r3.total:2d} "
                  f"way3[clr={r3.cleared:2d} {'OK' if r3.success else '..'} "
                  f"t={r3.virtual_time_s:7.1f}] "
                  f"way4[clr={r4.cleared:2d} {'OK' if r4.success else '..'} "
                  f"t={r4.virtual_time_s:7.1f} steps={r4.steps}]{flag}")
            if r4.error:
                print(f"           way4 error: {r4.error}")

    s3 = summarize(rows, "way3")
    s4 = summarize(rows, "way4")
    print("\n-- summary --")
    print(_fmt_summary(s3))
    print(_fmt_summary(s4))

    # head-to-head speed on episodes BOTH stacks cleared (the only fair time set)
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r.seed, {})[r.stack] = r
    both = [(d["way3"], d["way4"]) for d in by_seed.values()
            if d["way3"].success and d["way4"].success]
    if both:
        t3 = statistics.fmean(a.virtual_time_s for a, _ in both)
        t4 = statistics.fmean(b.virtual_time_s for _, b in both)
        print(f"\n-- both-cleared ({len(both)} episodes) mean virtual time --")
        print(f"  way3 {t3:8.1f} s   way4 {t4:8.1f} s   "
              f"ratio way4/way3 = {t4 / t3 if t3 else float('nan'):.3f}")

    # -- M7 gate --
    print("\n-- M7 acceptance gate (禁止10: Way4 full-clear >= Way3 full-clear) --")
    ok = s4.success_rate >= s3.success_rate - 1e-12
    if regressions:
        print(f"  Way4 FAILED on {len(regressions)} seed(s) Way3 cleared: {regressions}")
    verdict = "PASS" if ok else "FAIL"
    print(f"  Way3={s3.success_rate * 100:.2f}%  Way4={s4.success_rate * 100:.2f}%  "
          f"=> {verdict}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Way3 vs Way4 head-to-head (M7)")
    ap.add_argument("--problem", type=int, default=3, choices=(3, 4))
    ap.add_argument("-n", "--n", type=int, default=50, help="number of seeds")
    ap.add_argument("--seed", type=int, default=1000, help="base seed")
    ap.add_argument("--field", default="smooth", choices=("smooth", "constant"),
                    help="error field (constant = worst-case ±1°)")
    ap.add_argument("--max-steps", type=int, default=3000, help="Way4 macro-step cap")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every seed")
    args = ap.parse_args(argv)
    return compare(args.n, args.seed, args.problem, args.field, args.max_steps,
                   verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
