"""Run Way4 against the bundled offline simulator and save step traces."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Way4" / "src"))

from offline_sim.case import generate_case
from offline_sim.engine import Engine
from way4.core.observation import Observation
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline


class OfflineAdapter(Way3EngineAdapter):
    """Translate bundled offline_sim raw outcomes into Way4 observations."""
    def measure(self, x, y, channel):
        resp, out = self.engine.measure(x, y, channel)
        if out is None:
            return None, float(resp.get("virtual_time_s", self.engine.virtual_time_s))
        t = float(resp["virtual_time_s"])
        if out.result == "no_signal":
            obs = Observation.no_signal(time=t)
        elif out.result == "near":
            obs = Observation.near(time=t)
        elif out.result == "direction":
            obs = Observation.bearing(float(out.svd_deg), time=t)
        else:
            raise ValueError(out.result)
        return obs, t


def run(seed: int, max_steps: int = 120):
    case = generate_case(seed=seed, problem=3, field_kind="smooth", mode="formal")
    engine = Engine(case)
    engine.enter()
    pipe = Way4Pipeline(OfflineAdapter(engine), n_channels=20, max_steps=max_steps)
    rows = []
    original = pipe.executor.execute

    def traced(macro, state, budget):
        before = pipe._progress_key()
        result = original(macro, state, budget)
        rows.append({
            "step": pipe._steps,
            "action": macro.action_type.value,
            "target": list(macro.target),
            "scan_channels": list(macro.scan_channels),
            "clear_channel": macro.clear_channel,
            "certificate_gain": macro.certificate_gain,
            "state_before": {"x": state.x, "y": state.y, "channel": state.channel, "virtual_time_s": state.vt_us / 1e6},
            "primitives": [{"kind": p.primitive.kind.value, "channel": p.primitive.channel, "observation": p.observation.kind.value if p.observation else None, "cleared": p.cleared, "virtual_time_s": p.virtual_time_s} for p in result.primitives],
            "state_after": {"x": result.state.x, "y": result.state.y, "channel": result.state.channel, "virtual_time_s": result.state.vt_us / 1e6},
            "progress_key_before": repr(before),
        })
        return result

    pipe.executor.execute = traced
    result = pipe.run()
    return {"seed": seed, "result": result.__dict__, "trace": rows,
            "truth": [{"channel": j.channel, "x": j.x, "y": j.y, "r_eff": j.r_eff, "kind": j.kind} for j in case.jammers]}


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Way4" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in (sys.argv[2:] or [1000, 1001, 1002])]
    for seed in seeds:
        payload = run(seed)
        path = out_dir / f"way4_detailed_trace_seed_{seed}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"seed": seed, **payload["result"], "trace_file": str(path)}, ensure_ascii=False))
