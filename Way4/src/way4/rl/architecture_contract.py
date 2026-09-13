"""Machine-readable A01/A02 boundary for the learning layer."""

OBJECTIVE = "min_total_time_under_full_clear_hard_constraint"
SAFETY_OWNER = "math_safety_and_executor"
POLICY_ROLE = "rank_prevalidated_safe_candidates"
POLICY_OUTPUT = "candidate_index"
HARD_CONSTRAINTS = ("full_clear", "illegal_clear", "safety_violation")


def contract() -> dict:
    return {
        "objective": OBJECTIVE,
        "safety_owner": SAFETY_OWNER,
        "policy_role": POLICY_ROLE,
        "policy_output": POLICY_OUTPUT,
        "hard_constraints": list(HARD_CONSTRAINTS),
    }


def validate_policy_output(index, candidate_count: int) -> int:
    """PPO may select an index only; coordinates cannot cross this boundary."""
    if candidate_count < 1 or not isinstance(index, int) or not 0 <= index < candidate_count:
        raise ValueError("policy output must be a valid safe-candidate index")
    return index
