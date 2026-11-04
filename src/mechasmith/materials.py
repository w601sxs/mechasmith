"""Materials, including a simple model for 3D-printed parts.

Solid-volume mass is wrong for FDM prints: a 20 % infill part weighs far less
than its solid volume suggests. We approximate printed mass as
    shell (surface area x wall thickness, capped at the volume) at full density
  + the remaining interior at `infill` density.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    density: float              # kg/m^3 (solid)
    printed: bool = False
    infill: float = 1.0         # 0..1, interior fill fraction for printed parts
    wall_mm: float = 1.2        # perimeter + top/bottom skin thickness

    def effective_mass(self, volume_mm3: float, area_mm2: float) -> float:
        rho = self.density * 1e-9  # kg/mm^3
        if not self.printed:
            return rho * volume_mm3
        shell = min(volume_mm3, 0.5 * area_mm2 * self.wall_mm)  # area counts both faces of thin walls
        return rho * (shell + self.infill * (volume_mm3 - shell))


PLA = Material("PLA", 1240, printed=True, infill=0.2, wall_mm=1.2)
PLA_SOLID = Material("PLA (100%)", 1240, printed=True, infill=1.0)
TPU = Material("TPU", 1210, printed=True, infill=0.3, wall_mm=1.2)
ALUMINIUM = Material("Aluminium 6061", 2700)
STEEL = Material("Steel", 7850)
BOUGHT = Material("bought part (mass override)", 1000)  # density unused when mass_override set
