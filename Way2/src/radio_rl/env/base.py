"""The environment contract.

Both the self-contained :class:`~radio_rl.env.local_env.LocalEnv` and the
official HTTP client must return *exactly* the same :class:`DetectionObservation`
structure and obey the same timing semantics, so nothing downstream — belief,
candidates, agent — can tell which one it is talking to.

Iron rule (architecture principle A): the environment layer never imports models
or torch. It speaks only in the plain-data types from ``core.datatypes``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.constants import CONSTANTS
from ..core.datatypes import Action, DetectionObservation, ObservationType


class RadioEnv(ABC):
    """Abstract radio-localization environment.

    Concrete envs must set ``max_virtual_duration_s`` / ``max_real_duration_s``
    (populated at reset, mirroring the ``/enter`` response) and implement the
    four abstract members below.
    """

    max_virtual_duration_s: float = CONSTANTS.max_virtual_duration_s
    max_real_duration_s: float = CONSTANTS.program_time_limit_s

    @abstractmethod
    def reset(self, seed: int | None = None) -> DetectionObservation:
        """Begin a new episode; return the RESET observation carrying the
        initial pose (0, 0) on channel 1."""

    @abstractmethod
    def execute(self, action: Action) -> DetectionObservation:
        """Execute one action and return its observation. Handles SCAN, CLEAR
        and EXIT. After EXIT (or any deadline) ``finished`` becomes True."""

    @property
    @abstractmethod
    def finished(self) -> bool:
        """True once the episode has ended (exit or a deadline was hit)."""

    @property
    @abstractmethod
    def virtual_time_s(self) -> float:
        """Accumulated task (virtual) time — the quantity we minimise."""

    @abstractmethod
    def remaining_real_duration_s(self) -> float:
        """Program wall-clock budget still available (drives the emergency
        fallback). Local envs may report a simulated budget."""

    # -- shared helper ------------------------------------------------------
    @staticmethod
    def _reset_observation() -> DetectionObservation:
        return DetectionObservation(
            result_type=ObservationType.RESET,
            channel=CONSTANTS.initial_channel,
            position_x=CONSTANTS.initial_x,
            position_y=CONSTANTS.initial_y,
            virtual_time_delta=0.0,
        )
