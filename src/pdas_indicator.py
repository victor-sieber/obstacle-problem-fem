"""Measure a fixed-mesh reduction using the existing FEM indicator."""

import numpy as np


class PDASIndicator:
    def __init__(self, evaluate, relative_tolerance):
        if relative_tolerance is not None and not 0.0 < relative_tolerance < 1.0:
            raise ValueError("PDAS relative tolerance must be between zero and one")
        self.evaluate = evaluate
        self.relative_tolerance = relative_tolerance
        self.history = []
        self.indicator = None

    def __call__(self, gap, multiplier):
        self.indicator = self.evaluate(gap, multiplier)
        eta = float(np.linalg.norm(self.indicator["eta"]))
        if not np.isfinite(eta):
            raise RuntimeError("Nonfinite indicator during PDAS")
        self.history.append(eta)
        first = self.history[0]
        return (self.relative_tolerance is not None and first > 0.0
                and eta <= self.relative_tolerance * first)

    @property
    def ratio(self):
        first = self.history[0]
        return self.history[-1] / first if first > 0.0 else None


def passes_nodal_kkt(gap, multiplier, active, tolerance=1e-8):
    """Check the same absolute nodal conditions used in run validation."""
    return (
        np.all(gap >= -tolerance)
        and np.all(multiplier[active] >= -tolerance)
        and np.all(np.abs(multiplier[~active]) <= tolerance)
        and np.all(np.abs(gap * multiplier) <= tolerance)
    )
