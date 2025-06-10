from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass, field

@dataclass
class DrawOptions:
    color: Tuple[float, float, float] = field(
        default_factory=lambda: (0.5, 0.5, 0.5),
    )
    radius: float = field(default=0.4)
    length: float = field(default=1.6)
    opacity: float = field(default=1.0)

    @classmethod
    def from_dict(cls, opts: dict) -> DrawOptions:
        return cls(
            color=opts.get("color", (0.5, 0.5, 0.5)),
            radius=opts.get("radius", 0.4),
            length=opts.get("length", 1.6),
            opacity=opts.get("opacity", 1.0),
        )