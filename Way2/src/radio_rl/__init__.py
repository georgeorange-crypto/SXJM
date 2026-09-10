"""radio_rl — modular DRL for radio interference source localization (Way 2).

The one and only main pipeline is::

    Simulator -> Math State Estimator -> Candidate Generator
              -> Feature Builder -> Learnable Agent -> Safety Shield -> Action

Design rule that must never be violated:

    The neural network never mutates the mathematical Belief.
    It only *scores* pre-generated candidate actions given the math-derived state.

See README.md for the full architecture and the immovable-core / pluggable split.
"""

__version__ = "0.1.0"
