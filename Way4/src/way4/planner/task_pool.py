"""Planner-owned remaining-task and WAIT_FOR_ROUTE bookkeeping.

Belief statuses remain authoritative; waiting is an execution preference only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from ..belief import ChannelStatus
from .opportunities import RemainingTask, insertion_cost


@dataclass
class WaitingTask:
    task: RemainingTask
    wait_age: int = 0
    skipped_count: int = 0
    current_best_dedicated_cost: float = float("inf")
    next_route_opportunity_cost: float = float("inf")
    assigned_waypoint: Optional[tuple[float, float]] = None


class RemainingTaskPool:
    """Small deterministic task registry used by receding-horizon planners."""

    def __init__(self, starvation_limit: int = 8) -> None:
        self.starvation_limit = int(starvation_limit)
        self.waiting: Dict[int, WaitingTask] = {}
        self.pending_tasks: Dict[str, list[RemainingTask]] = {
            kind: [] for kind in ("EXPLORE", "INITIALIZE", "REFINE", "REACQUIRE",
                                  "PURSUE", "CLEAR", "VERIFY", "BACKBONE")
        }

    def sync(self, belief) -> None:
        """Create/remove refine tasks without changing any belief status."""
        self.pending_tasks = {kind: [] for kind in self.pending_tasks}
        active = set()
        for channel, b in belief.channels.items():
            status = b.status
            if status in (ChannelStatus.UNKNOWN, ChannelStatus.PRESENT_UNOBSERVED):
                self.pending_tasks["EXPLORE"].append(RemainingTask("EXPLORE", channel=int(channel)))
                self.pending_tasks["VERIFY"].append(RemainingTask("VERIFY", channel=int(channel)))
                self.pending_tasks["BACKBONE"].append(RemainingTask("BACKBONE", channel=int(channel)))
            if status == ChannelStatus.PRESENT_UNOBSERVED:
                self.pending_tasks["INITIALIZE"].append(RemainingTask("INITIALIZE", channel=int(channel)))
            if status in (ChannelStatus.DETECTED, ChannelStatus.INITIALIZED):
                active.add(int(channel))
                self.waiting.setdefault(
                    int(channel), WaitingTask(RemainingTask("REFINE", channel=int(channel), mandatory=True))
                )
                self.pending_tasks["REFINE"].append(RemainingTask("REFINE", channel=int(channel)))
                self.pending_tasks["REACQUIRE"].append(RemainingTask("REACQUIRE", channel=int(channel)))
                self.pending_tasks["PURSUE"].append(RemainingTask("PURSUE", channel=int(channel)))
            if getattr(b, "is_clearable", False):
                self.pending_tasks["CLEAR"].append(RemainingTask("CLEAR", channel=int(channel)))
        for channel in list(self.waiting):
            if channel not in active:
                del self.waiting[channel]

    def register(self, task: RemainingTask) -> None:
        """Add an externally discovered task without touching belief state."""
        key = str(task.task_type).upper()
        if key not in self.pending_tasks:
            raise ValueError(f"unsupported task type: {task.task_type}")
        self.pending_tasks[key].append(task)

    def consider_wait(self, channel: int, dedicated_cost: float,
                      route_cost: float, waypoint, *, delay_safe: bool = True) -> bool:
        """Wait iff a viable route opportunity is cheaper and starvation is safe."""
        item = self.waiting.get(int(channel))
        if item is None:
            return False
        item.current_best_dedicated_cost = float(dedicated_cost)
        item.next_route_opportunity_cost = float(route_cost)
        if item.skipped_count >= self.starvation_limit or not delay_safe:
            return False
        if float(route_cost) < float(dedicated_cost):
            item.skipped_count += 1
            item.wait_age += 1
            item.assigned_waypoint = (float(waypoint[0]), float(waypoint[1]))
            return True
        return False

    def mark_served(self, channel: int) -> None:
        item = self.waiting.get(int(channel))
        if item:
            item.wait_age = 0
            item.skipped_count = 0
            item.assigned_waypoint = None

    def assign_route_opportunities(self, route, dedicated_costs: Dict[int, float],
                                   *, viable=lambda _c, _p: True,
                                   delay_safe=lambda _c, _p: True) -> dict:
        """Bind waiting channels to the cheapest viable future route waypoint."""
        points = list(route)
        assignments = {}
        for channel in list(self.waiting):
            options = [(insertion_cost(points, p), p) for p in points[1:]
                       if viable(channel, p)]
            if not options:
                continue
            route_cost, point = min(options, key=lambda x: (x[0], x[1]))
            if self.consider_wait(channel, dedicated_costs.get(channel, float("inf")),
                                  route_cost, point,
                                  delay_safe=bool(delay_safe(channel, point))):
                assignments[channel] = point
        return assignments

    def snapshot(self) -> dict:
        return {str(c): {
            "status": "WAITING_FOR_ROUTE",
            "wait_age": x.wait_age,
            "skipped_count": x.skipped_count,
            "current_best_dedicated_cost": x.current_best_dedicated_cost,
            "next_route_opportunity_cost": x.next_route_opportunity_cost,
            "assigned_waypoint": x.assigned_waypoint,
        } for c, x in self.waiting.items()}
