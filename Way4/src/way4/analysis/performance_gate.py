"""Safety-first engineering performance gates (AI01--AI04)."""


GATES = {"G1": 1000.0, "G2": 700.0, "G3": 500.0, "FINAL": 400.0}


def performance_gate(*, mean_time_s_per_target: float, full_clear_rate: float,
                     clear_correct: bool, certificate_sound: bool, gate: str = "G1") -> dict:
    """Evaluate a target-time gate; any correctness regression fails it."""
    key = str(gate).upper()
    if key not in GATES:
        raise ValueError(f"unknown performance gate: {gate}")
    t, rate = float(mean_time_s_per_target), float(full_clear_rate)
    if t < 0 or not 0.0 <= rate <= 1.0:
        raise ValueError("time must be non-negative and full-clear rate in [0,1]")
    time_ok = t < GATES[key] if key != "FINAL" else t <= GATES[key]
    passed = time_ok and rate == 1.0 and bool(clear_correct) and bool(certificate_sound)
    return {"gate": key, "threshold_s": GATES[key], "time_ok": time_ok,
            "full_clear_ok": rate == 1.0, "clear_correct": bool(clear_correct),
            "certificate_sound": bool(certificate_sound), "passed": passed}
