"""Candidate layer — enumerate legal, annotated action options.

The generator is the single source of concrete actions (principle C). It is
torch-free and depends only on the belief and the world model.
"""

from __future__ import annotations

from typing import Any, Optional

from ..world_model.base import WorldModel
from .generator import CandidateGenerator, CandidateSet


def build_candidate_generator(cfg: Any = None,
                              world_model: Optional[WorldModel] = None
                              ) -> CandidateGenerator:
    return CandidateGenerator(cfg=cfg, world_model=world_model)


__all__ = ["CandidateGenerator", "CandidateSet", "build_candidate_generator"]
