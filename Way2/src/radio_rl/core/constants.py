"""Frozen problem constants (CUMCM 2026 Problem B).

Every hard rule from the problem statement lives here and *only* here. No other
module is allowed to hard-code a magic number such as ``if distance < 20``.
Always reference ``CONSTANTS.optical_radius`` instead.

Units: metres, seconds, degrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians


@dataclass(frozen=True)
class TaskConstants:
    # --- Field geometry ----------------------------------------------------
    region_radius: float = 1800.0          # circular operating area radius (m)

    # --- Channels ----------------------------------------------------------
    channel_min: int = 1
    channel_max: int = 20                  # 20 mutually-distinct channels
    # One interference source occupies one channel; a channel has <= 1 source.

    # --- Source population (problems 3 & 4) --------------------------------
    source_count_min: int = 10
    source_count_max: int = 16

    # --- Signal reception --------------------------------------------------
    receive_radius_min: float = 1000.0     # effective receive radius R in
    receive_radius_max: float = 1500.0     # [1000, 1500] m (unknown per source)

    # --- Direction finding -------------------------------------------------
    bearing_error_deg: float = 1.0         # |bearing error| <= 1 deg,
    #   *fixed for a given location*, varying statistically across locations.

    too_strong_radius: float = 5.0         # <= 5 m: signal "too strong", no bearing
    clear_radius: float = 20.0             # <= 20 m from source: a clear succeeds.
    #   The problem's "optical positioning within 20 m" is realised by the
    #   environment as this clear-success radius: there is NO separate optical
    #   action and the robot is NEVER told the true source coordinates. Getting
    #   within 20 m and issuing CLEAR is the whole mechanism.

    # --- Problem 4: directional (180-degree) sources -----------------------
    directional_half_angle_deg: float = 90.0   # source radiates into a 180 deg arc

    # --- Kinematics & action durations ------------------------------------
    robot_speed: float = 5.0               # m/s
    channel_switch_time: float = 1.0       # s per channel switch (only if changed)
    detection_time: float = 5.0            # s per direction-finding detection
    clear_hit_time: float = 5.0            # s for a CLEAR that neutralises a source
    clear_miss_time: float = 3.0           # s for a CLEAR that hits nothing

    # --- Initial robot state ----------------------------------------------
    initial_x: float = 0.0
    initial_y: float = 0.0
    initial_channel: int = 1

    # --- Coordinate sanity bound (matches the official engine) -------------
    coord_abs_max: float = 2.0e6           # |x|,|y| must stay within this

    # --- Task-time envelope (virtual time — the quantity we minimise) ------
    max_virtual_duration_s: float = 360000.0   # hard cap on accumulated task time

    # --- Competition wall-clock envelope (program run time, NOT task time) -
    program_time_limit_s: float = 1200.0   # 20 min hard cap after /enter
    test_window_s: float = 1500.0          # 25 min overall test window
    emergency_margin_s: float = 45.0       # switch to math fallback below this

    # --- Convenience -------------------------------------------------------
    @property
    def num_channels(self) -> int:
        return self.channel_max - self.channel_min + 1

    @property
    def bearing_error_rad(self) -> float:
        return radians(self.bearing_error_deg)

    @property
    def directional_half_angle_rad(self) -> float:
        return radians(self.directional_half_angle_deg)

    @property
    def diameter(self) -> float:
        return 2.0 * self.region_radius

    def channel_index(self, channel: int) -> int:
        """Map a 1-based channel number to a 0-based array index."""
        return channel - self.channel_min

    def channel_number(self, index: int) -> int:
        """Map a 0-based array index back to a 1-based channel number."""
        return index + self.channel_min


# The single shared instance. Import this, do not instantiate ad-hoc copies.
CONSTANTS = TaskConstants()
