"""A minimal Hydra-compatible config composer built on OmegaConf.

We deliberately do *not* depend on Hydra (it is not installed and pulls in a
lot of runtime machinery). Instead we support the small subset that keeps
configs modular and swappable by editing one line:

* a root ``config.yaml`` with a ``defaults:`` list of ``{group: option}`` items
  (plus the literal ``_self_`` marking where the root's own keys merge);
* one YAML file per option inside ``configs/<group>/<option>.yaml``;
* command-line overrides in dotlist form, either selecting a different option
  for a group (``algorithm=ppo``) or setting a value (``training.lr=3e-4``).

The whole point of the architecture is that switching PPO+MLP to, say,
PPO+Transformer is a single ``channel_encoder=transformer`` override and nothing
in the code changes. This composer is what makes that true.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf


def _config_root() -> Path:
    """Absolute path to the packaged ``configs/`` directory.

    Layout: ``src/radio_rl/core/config.py`` -> ``configs`` sits next to
    ``src`` at the project root, i.e. three parents up then ``configs``.
    """
    return Path(__file__).resolve().parents[3] / "configs"


def _known_groups(config_dir: Path) -> set[str]:
    """Group names are exactly the sub-directories of ``configs/``."""
    return {p.name for p in config_dir.iterdir() if p.is_dir()}


def _split_overrides(
    overrides: Iterable[str], groups: set[str]
) -> tuple[dict[str, str], list[str]]:
    """Partition ``key=value`` tokens into group selections and value dotlists.

    A token whose key (before the first dot / equals) names a known group and
    has no dotted path selects that group's option; everything else is an
    OmegaConf dotlist value override applied at the end.
    """
    group_choice: dict[str, str] = {}
    value_dotlist: list[str] = []
    for tok in overrides:
        if "=" not in tok:
            raise ValueError(f"malformed override '{tok}' (expected key=value)")
        key, value = tok.split("=", 1)
        key = key.strip()
        if "." not in key and key in groups:
            group_choice[key] = value.strip()
        else:
            value_dotlist.append(tok)
    return group_choice, value_dotlist


def _parse_defaults(
    defaults: Sequence | None,
) -> tuple[list[tuple[str, str]], int]:
    """Return ``[(group, option), ...]`` and the index of the ``_self_`` marker.

    ``_self_`` defaults to the *end* (root keys win) when omitted, matching the
    most common Hydra usage.
    """
    parsed: list[tuple[str, str]] = []
    self_index = -1
    if defaults is None:
        return parsed, self_index
    for i, item in enumerate(defaults):
        if item == "_self_":
            self_index = len(parsed)
            continue
        if isinstance(item, (dict, DictConfig)):
            for group, option in item.items():
                parsed.append((str(group), str(option)))
        else:
            raise ValueError(
                f"unsupported defaults entry {item!r}; use {{group: option}} or _self_"
            )
    if self_index == -1:
        self_index = len(parsed)  # root merges last by default
    return parsed, self_index


def compose(
    config_name: str = "config",
    overrides: Sequence[str] | None = None,
    config_dir: Path | str | None = None,
) -> DictConfig:
    """Compose the full config tree.

    Args:
        config_name: root file stem inside ``configs/`` (default ``config``).
        overrides: CLI-style ``key=value`` tokens (group selections + values).
        config_dir: override the packaged ``configs/`` location (for tests).
    """
    root_dir = Path(config_dir) if config_dir is not None else _config_root()
    overrides = list(overrides or [])
    groups = _known_groups(root_dir)
    group_choice, value_dotlist = _split_overrides(overrides, groups)

    root = OmegaConf.load(root_dir / f"{config_name}.yaml")
    if not isinstance(root, DictConfig):
        raise TypeError(f"{config_name}.yaml must be a mapping")

    raw_defaults = root.pop("defaults", None)
    defaults_list = None if raw_defaults is None else list(raw_defaults)
    parsed_defaults, self_index = _parse_defaults(defaults_list)

    cfg = OmegaConf.create({})
    for pos, (group, option) in enumerate(parsed_defaults):
        if pos == self_index:
            cfg = OmegaConf.merge(cfg, root)
        chosen = group_choice.pop(group, option)
        option_file = root_dir / group / f"{chosen}.yaml"
        if not option_file.is_file():
            raise FileNotFoundError(
                f"config group '{group}' has no option '{chosen}' "
                f"(looked for {option_file})"
            )
        loaded = OmegaConf.load(option_file)
        cfg = OmegaConf.merge(cfg, OmegaConf.create({group: loaded}))
    if self_index >= len(parsed_defaults):
        cfg = OmegaConf.merge(cfg, root)

    # A group override for a group not listed in defaults still loads it.
    for group, chosen in group_choice.items():
        option_file = root_dir / group / f"{chosen}.yaml"
        if not option_file.is_file():
            raise FileNotFoundError(
                f"config group '{group}' has no option '{chosen}' "
                f"(looked for {option_file})"
            )
        loaded = OmegaConf.load(option_file)
        cfg = OmegaConf.merge(cfg, OmegaConf.create({group: loaded}))

    if value_dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(value_dotlist))

    assert isinstance(cfg, DictConfig)
    return cfg


def to_container(cfg: DictConfig | ListConfig) -> object:
    """Resolve interpolations and return plain Python containers."""
    return OmegaConf.to_container(cfg, resolve=True)
