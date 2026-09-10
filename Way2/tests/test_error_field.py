"""Error fields: bearing error stays in [-1, 1] deg and is a deterministic
function of position (never per-measurement RNG), for every registered kind."""

from __future__ import annotations

import pytest

from radio_rl.env.error_field import FIELD_KINDS, make_error_field

_POINTS = [(0, 0), (100, 50), (-200, 300), (500, -400), (1700, 0),
           (-1200, -800), (33.3, -12.7), (999.9, 999.9)]


@pytest.mark.parametrize("kind", FIELD_KINDS)
def test_error_bounded(kind):
    f = make_error_field(kind, seed=3)
    for (x, y) in _POINTS:
        e = f.error_deg(x, y)
        assert -1.0 - 1e-9 <= e <= 1.0 + 1e-9, f"{kind} out of range at {(x, y)}: {e}"


@pytest.mark.parametrize("kind", FIELD_KINDS)
def test_error_deterministic(kind):
    f = make_error_field(kind, seed=7)
    for (x, y) in _POINTS:
        assert f.error_deg(x, y) == f.error_deg(x, y)  # repeatable within instance


@pytest.mark.parametrize("kind", FIELD_KINDS)
def test_error_reproducible_across_instances(kind):
    a = make_error_field(kind, seed=42)
    b = make_error_field(kind, seed=42)
    for (x, y) in _POINTS:
        assert a.error_deg(x, y) == pytest.approx(b.error_deg(x, y), abs=1e-12)


def test_kinds_registered():
    for expected in ("smooth", "iid", "biased", "adversarial", "piecewise"):
        assert expected in FIELD_KINDS
