"""Tests for the run-visualization observer (recorder + HTML report).

The viz layer must stay a *faithful, read-only* view of an episode: recording a
run may not change its outcome, every frame must be JSON-serializable, and the
rendered page must be self-contained (data embedded, no leftover template
placeholders, JSON that round-trips).
"""

from __future__ import annotations

import json
import re

import pytest

from radio_rl.core.config import compose
from radio_rl.viz import record_run, record_runs, render_html, write_report


def _cfg(problem: int = 3, algo: str = "heuristic"):
    return compose(overrides=[f"problem={problem}", f"algorithm={algo}"])


def test_record_run_captures_frames_and_truth():
    rec = record_run(_cfg(problem=3), seed=7, max_steps=2000)
    # default label is the registered agent name; an explicit label (as
    # visualize.py passes) would override it.
    assert rec.label == "greedy_math"
    rec_labeled = record_run(_cfg(problem=3), seed=7, label="heuristic", max_steps=2000)
    assert rec_labeled.label == "heuristic"
    assert rec.problem == 3
    assert rec.frames, "expected at least one recorded frame"
    # practice-mode ground truth is present and well-formed
    assert rec.truth is not None
    assert 10 <= rec.truth["total"] <= 16
    assert len(rec.truth["jammers"]) == rec.truth["total"]
    for j in rec.truth["jammers"]:
        assert j["kind"] in ("omni", "dir")
        assert 1 <= j["channel"] <= 20
    # P3 => all omnidirectional
    assert rec.truth["n_dir"] == 0


def test_every_frame_is_json_serializable_and_shaped():
    rec = record_run(_cfg(), seed=3, max_steps=2000)
    blob = json.dumps(rec.to_dict())          # must not raise
    assert blob
    for fr in rec.frames:
        assert fr["type"] in ("scan", "clear", "exit")
        assert fr["result"] in (
            "signal", "no_signal", "near", "clear_success", "clear_failure", "reset",
        )
        assert len(fr["pose"]) == 2
        assert isinstance(fr["vt"], (int, float))
        # a bearing is present exactly when a scan returned a signal
        if fr["type"] == "scan" and fr["result"] == "signal":
            assert "bearing" in fr and 0.0 <= fr["bearing"] < 360.0


def test_recording_does_not_change_the_outcome():
    """The traced run and a plain run on the same seed must agree bit-for-bit on
    the metrics that matter — recording is a pure observer."""
    from radio_rl.pipeline import Pipeline

    cfg = _cfg(problem=3)
    rec = record_run(cfg, seed=11, max_steps=2000)

    stats, _ = Pipeline(cfg).run_episode(seed=11, max_steps=2000, collect_trace=False)
    assert rec.summary["sources_cleared"] == stats.sources_cleared
    assert rec.summary["sources_total"] == stats.sources_total
    # summary rounds virtual time to 3 decimals for display; compare at that tol
    assert rec.summary["virtual_time_s"] == pytest.approx(stats.virtual_time, abs=5e-4)


def test_render_html_is_self_contained(tmp_path):
    recs = [record_run(_cfg(), seed=7, max_steps=2000)]
    html = render_html(recs)
    assert html.startswith("<!doctype html>")
    # no template placeholders survive
    assert "__DATA__" not in html and "__TITLE__" not in html
    # the embedded JSON round-trips after undoing the </-escaping
    m = re.search(r'(?s)<script type="application/json" id="data">(.*?)</script>', html)
    assert m
    payload = json.loads(m.group(1).replace(r"<\/", "</"))
    assert payload["runs"][0]["frames"]


def test_write_report_writes_file(tmp_path):
    recs = record_runs(
        [{"cfg": _cfg(algo="heuristic"), "label": "greedy"}], seed=5, max_steps=2000
    )
    out = write_report(recs, tmp_path / "sub" / "view.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "canvas" in text and "greedy" in text


def test_multiple_runs_share_layout_when_seed_matches():
    """Same seed => identical jammer layout across algorithms (fair comparison)."""
    recs = record_runs(
        [
            {"cfg": _cfg(algo="heuristic"), "label": "a"},
            {"cfg": _cfg(algo="heuristic"), "label": "b"},
        ],
        seed=21, max_steps=2000,
    )
    ja = sorted((j["channel"], round(j["x"], 6), round(j["y"], 6))
                for j in recs[0].truth["jammers"])
    jb = sorted((j["channel"], round(j["x"], 6), round(j["y"], 6))
                for j in recs[1].truth["jammers"])
    assert ja == jb
