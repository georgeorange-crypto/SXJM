"""Task commitment state used to prevent Search/Refine oscillation (K01)."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class FocusState:
    channel: Optional[int] = None
    score: float = 0.0

    @property
    def active(self) -> bool:
        return self.channel is not None


class FocusController:
    """Deterministic FOCUS selector based on completion probability per second."""
    def __init__(self, threshold: float = 1e-3):
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        self.threshold = float(threshold)
        self.state = FocusState()

    def consider(self, channel: int, p_completion: float,
                 expected_to_finish_s: float) -> bool:
        if not 0.0 <= float(p_completion) <= 1.0:
            raise ValueError("p_completion must be in [0, 1]")
        if expected_to_finish_s <= 0.0:
            raise ValueError("expected_to_finish_s must be positive")
        score = float(p_completion) / float(expected_to_finish_s)
        if score >= self.threshold and (not self.state.active or score > self.state.score):
            self.state = FocusState(int(channel), score)
            return True
        return False

    def may_exit(self, *, no_progress: bool = False, hypothesis_invalid: bool = False,
                 reacquire_failed: bool = False, better_task: bool = False,
                 watchdog: bool = False) -> bool:
        return any((no_progress, hypothesis_invalid, reacquire_failed, better_task, watchdog))

    def clear(self) -> None:
        self.state = FocusState()
