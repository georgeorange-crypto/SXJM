"""Environment layer.

Two interchangeable implementations of :class:`RadioEnv`:

* :class:`LocalEnv` — a self-contained, network-free physics/timing oracle
  (the default, used for training and fast evaluation);
* :class:`OfficialEnv` — an HTTP client for the official competition simulator.

``build_env(cfg)`` picks one from config so ``env=official`` is a one-line switch.
This package never imports models or torch (architecture principle A).
"""

from __future__ import annotations

from typing import Any

from .base import RadioEnv
from .local_env import LocalEnv
from .official_env import OfficialEnv


def _to_dict(node: Any) -> dict:
    if node is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(node):
            return dict(OmegaConf.to_container(node, resolve=True))  # type: ignore[arg-type]
    except Exception:
        pass
    return dict(node)


def build_env(cfg) -> RadioEnv:
    """Construct the environment selected by ``cfg.env.type``."""
    env_type = str(cfg.env.type).lower()
    if env_type == "local":
        ef = _to_dict(cfg.get("error_field"))
        return LocalEnv(
            problem=int(cfg.get("problem", 3)),
            error_field_kind=ef.get("kind", "smooth"),
            error_field_params=ef.get("params") or {},
            generator=_to_dict(cfg.get("generator")),
            max_virtual_duration_s=float(cfg.env.get("max_virtual_duration_s", 360000.0)),
            max_real_duration_s=float(cfg.env.get("max_real_duration_s", 1200.0)),
        )
    if env_type == "official":
        return OfficialEnv(
            base_url=str(cfg.env.get("base_url", "http://127.0.0.1:2026")),
            robot_id=str(cfg.env.get("robot_id", "default")),
            timeout_s=float(cfg.env.get("timeout_s", 10.0)),
            connect_retries=int(cfg.env.get("connect_retries", 30)),
            retry_delay_s=float(cfg.env.get("retry_delay_s", 0.5)),
        )
    raise ValueError(f"unknown env type {env_type!r} (expected 'local' or 'official')")


__all__ = ["RadioEnv", "LocalEnv", "OfficialEnv", "build_env"]
