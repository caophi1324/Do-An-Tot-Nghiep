"""One Euro Filter smoothing and coordinate transforms for bounding boxes."""
from __future__ import annotations

import math
from typing import Sequence, Tuple, Union
import numpy as np


class OneEuro:
    """One Euro Filter for responsive, jitter-free signal smoothing.
    
    Filters higher frequencies when signal is static to eliminate jitter,
    and dynamically reduces filtering as velocity increases to prevent lag.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None
        self.t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        if cutoff <= 0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update_params(
        self,
        min_cutoff: float | None = None,
        beta: float | None = None,
        d_cutoff: float | None = None,
    ) -> None:
        """Dynamically updates filter tuning parameters."""
        if min_cutoff is not None:
            self.min_cutoff = float(min_cutoff)
        if beta is not None:
            self.beta = float(beta)
        if d_cutoff is not None:
            self.d_cutoff = float(d_cutoff)

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x: Sequence[float] | np.ndarray, t: float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if self.t_prev is None:
            self.x_prev = x_arr.copy()
            self.dx_prev = np.zeros_like(x_arr)
            self.t_prev = float(t)
            return x_arr

        dt = max(float(t - self.t_prev), 1e-6)
        dx = (x_arr - self.x_prev) / dt

        # Estimate derivative using low-pass filter
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Adaptive cutoff frequency based on velocity
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        
        # Element-wise alpha for vectors
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-5))
        a = 1.0 / (1.0 + tau / dt)

        # Filtered output
        x_hat = a * x_arr + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = float(t)
        return x_hat


def xyxy_to_cxcywh(box: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=np.float64)


def cxcywh_to_xyxy(box: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = box[0], box[1], box[2], box[3]
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)


class BoxSmoother:
    """Smooths bounding boxes over time using One Euro filter in cx, cy, w, h space."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0) -> None:
        self.filter = OneEuro(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.last_box: np.ndarray | None = None
        self.last_time: float | None = None

    @property
    def x_smoother(self) -> OneEuro:
        """Alias to internal filter for backward compatibility."""
        return self.filter

    def update_params(
        self,
        min_cutoff: float | None = None,
        beta: float | None = None,
        d_cutoff: float | None = None,
    ) -> None:
        """Pushes updated smoothing parameters into active filter."""
        self.filter.update_params(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def update(self, box_xyxy: Sequence[float] | np.ndarray, timestamp: float) -> np.ndarray:
        cxcywh = xyxy_to_cxcywh(box_xyxy)
        smoothed_cxcywh = self.filter(cxcywh, timestamp)
        smoothed_xyxy = cxcywh_to_xyxy(smoothed_cxcywh)
        self.last_box = smoothed_xyxy
        self.last_time = timestamp
        return smoothed_xyxy

    def reset(self) -> None:
        self.filter.reset()
        self.last_box = None
        self.last_time = None
