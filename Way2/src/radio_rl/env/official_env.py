"""Official-simulator client — the same :class:`RadioEnv` contract over HTTP.

The official simulator is a Windows binary serving JSON on loopback
(``http://127.0.0.1:2026``). This client speaks exactly the wire protocol the
reference server enforces (POST to ``/enter`` ``/measure`` ``/clear`` ``/exit``
with ``arena_id`` / ``robot_id`` / ``request_id`` and, for measure/clear, a
``position`` object and integer ``channel``), and translates each response into
the *identical* :class:`DetectionObservation` the local env produces — so the
agent cannot tell the two apart.

Only the standard library is used (urllib), mirroring the reference client.
Robot position and current channel are tracked client-side (the server does not
echo them), consistent with the "last legal-action pose" timing model.
"""

from __future__ import annotations

import itertools
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..core.constants import CONSTANTS
from ..core.datatypes import (
    Action,
    ActionType,
    DetectionObservation,
    ObservationType,
)
from .base import RadioEnv

_MEASURE_RESULT_MAP = {
    "no_signal": ObservationType.NO_SIGNAL,
    "near": ObservationType.TOO_STRONG,
    "direction": ObservationType.SIGNAL,
}


class OfficialEnv(RadioEnv):
    """HTTP client for the official competition simulator."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:2026",
        robot_id: str = "default",
        timeout_s: float = 10.0,
        connect_retries: int = 30,
        retry_delay_s: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.timeout_s = float(timeout_s)
        self.connect_retries = int(connect_retries)
        self.retry_delay_s = float(retry_delay_s)

        self._seq = itertools.count(1)
        self._pos_x = CONSTANTS.initial_x
        self._pos_y = CONSTANTS.initial_y
        self._channel = CONSTANTS.initial_channel
        self._virtual_time_s = 0.0
        self._remaining_real_s = CONSTANTS.program_time_limit_s
        self._finished = False
        self._finish_reason: str | None = None

    # -- HTTP plumbing ------------------------------------------------------
    def _new_request_id(self, tag: str) -> str:
        return f"{tag}-{next(self._seq)}"

    def _base_payload(self, tag: str) -> dict:
        return {
            "arena_id": "default",
            "robot_id": self.robot_id,
            "request_id": self._new_request_id(tag),
        }

    def _post(self, path: str, payload: dict) -> tuple[int, dict | None]:
        """Return (http_status, json | None). A closed connection with no body
        (server finished / interface not open) is reported as (0, None)."""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode("utf-8"))
            except Exception:
                return e.code, None
        except (URLError, ConnectionError, OSError):
            return 0, None

    # -- lifecycle ----------------------------------------------------------
    def reset(self, seed: int | None = None) -> DetectionObservation:
        """seed is ignored: the official server owns the (hidden) case. Retries
        the /enter until the binary is listening and accepts."""
        self._seq = itertools.count(1)
        self._pos_x = CONSTANTS.initial_x
        self._pos_y = CONSTANTS.initial_y
        self._channel = CONSTANTS.initial_channel
        self._virtual_time_s = 0.0
        self._finished = False
        self._finish_reason = None

        last_status, body = 0, None
        for _ in range(max(1, self.connect_retries)):
            last_status, body = self._post("/enter", self._base_payload("enter"))
            if body is not None and body.get("accepted") is True:
                break
            time.sleep(self.retry_delay_s)
        if body is None or body.get("accepted") is not True:
            raise ConnectionError(
                f"could not /enter the official simulator at {self.base_url} "
                f"(last status {last_status}, body {body})"
            )
        self.max_virtual_duration_s = float(
            body.get("max_virtual_duration_s", CONSTANTS.max_virtual_duration_s)
        )
        self.max_real_duration_s = float(
            body.get("max_real_duration_s", CONSTANTS.program_time_limit_s)
        )
        self._remaining_real_s = float(
            body.get("remaining_real_duration_s", self.max_real_duration_s)
        )
        self._virtual_time_s = float(body.get("virtual_time_s", 0.0))
        return self._reset_observation()

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def finish_reason(self) -> str | None:
        return self._finish_reason

    @property
    def virtual_time_s(self) -> float:
        return self._virtual_time_s

    def remaining_real_duration_s(self) -> float:
        return self._remaining_real_s

    # -- action dispatch ----------------------------------------------------
    def execute(self, action: Action) -> DetectionObservation:
        if self._finished:
            return self._noop_observation(action)

        if action.action_type == ActionType.EXIT:
            self._post("/exit", self._base_payload("exit"))
            self._finish("user_exit")
            return DetectionObservation(
                result_type=ObservationType.NO_SIGNAL,
                channel=self._channel,
                position_x=self._pos_x,
                position_y=self._pos_y,
                virtual_time_delta=0.0,
            )

        if action.action_type == ActionType.SCAN:
            return self._request_action("/measure", "measure", action, is_clear=False)
        if action.action_type == ActionType.CLEAR:
            return self._request_action("/clear", "clear", action, is_clear=True)
        raise ValueError(f"unknown action type {action.action_type!r}")

    def _request_action(
        self, path: str, tag: str, action: Action, *, is_clear: bool
    ) -> DetectionObservation:
        payload = self._base_payload(tag)
        payload["position"] = {"x": action.target_x, "y": action.target_y}
        payload["channel"] = int(action.channel)
        status, body = self._post(path, payload)

        # Closed connection / missing body => the test window has ended.
        if body is None:
            self._finish("closed")
            return self._noop_observation(action)
        if body.get("accepted") is not True:
            # Malformed/duplicate request or interface refused it: nothing moved.
            return self._noop_observation(action)

        prev_vt = self._virtual_time_s
        self._virtual_time_s = float(body.get("virtual_time_s", prev_vt))
        delta = max(0.0, self._virtual_time_s - prev_vt)

        # Track pose client-side (server does not echo it).
        self._pos_x, self._pos_y = action.target_x, action.target_y
        if not is_clear:
            self._channel = int(action.channel)   # /clear does not switch channel

        if is_clear:
            hit = body.get("clear_result") == "success"
            return DetectionObservation(
                result_type=ObservationType.CLEAR_SUCCESS if hit else ObservationType.CLEAR_FAILURE,
                channel=int(action.channel),
                position_x=action.target_x,
                position_y=action.target_y,
                virtual_time_delta=delta,
                clear_success=bool(hit),
            )

        result = body.get("measure_result", "no_signal")
        obs_type = _MEASURE_RESULT_MAP.get(result, ObservationType.NO_SIGNAL)
        bearing = body.get("svd_deg") if obs_type == ObservationType.SIGNAL else None
        return DetectionObservation(
            result_type=obs_type,
            channel=int(action.channel),
            position_x=action.target_x,
            position_y=action.target_y,
            virtual_time_delta=delta,
            bearing_deg=(None if bearing is None else float(bearing)),
        )

    # -- misc ---------------------------------------------------------------
    def _finish(self, reason: str) -> None:
        self._finished = True
        self._finish_reason = reason

    def _noop_observation(self, action: Action) -> DetectionObservation:
        return DetectionObservation(
            result_type=ObservationType.NO_SIGNAL,
            channel=action.channel,
            position_x=self._pos_x,
            position_y=self._pos_y,
            virtual_time_delta=0.0,
        )
