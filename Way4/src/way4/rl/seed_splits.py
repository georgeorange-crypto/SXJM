"""Explicit non-overlapping development/train/validation/test seed splits."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedSplits:
    train: tuple[int, ...] = tuple(range(10000, 10200))
    val: tuple[int, ...] = tuple(range(11000, 11050))
    test: tuple[int, ...] = tuple(range(12000, 12050))
    development: tuple[int, ...] = tuple(range(2000, 2010))

    def validate(self):
        groups = {"train": self.train, "val": self.val, "test": self.test}
        seen = {}
        for name, seeds in groups.items():
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"duplicate seed in {name}")
            for seed in seeds:
                if seed in seen:
                    raise ValueError(f"seed overlap: {seed} ({seen[seed]}, {name})")
                seen[seed] = name
        if set(self.development) & set(seen):
            raise ValueError("development seeds overlap train/val/test")
        return True
