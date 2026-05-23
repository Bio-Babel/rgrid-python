"""Regression test: ``points_grob()`` no-arg default mirrors R.

R parity (primitives.R:1562-1563): ``pointsGrob`` defaults
``x = stats::runif(10), y = stats::runif(10)`` — 10 random points,
re-evaluated per call. The Python port mirrors this via
``np.random.uniform(0, 1, 10)`` using the numpy global RNG so users
can reproduce via ``np.random.seed(...)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from grid_py import points_grob


class TestPointsGrobDefault:
    def test_no_args_produces_10_points(self):
        """R: length(pointsGrob()$x) == 10."""
        g = points_grob()
        assert len(g.x) == 10
        assert len(g.y) == 10

    def test_consecutive_calls_differ(self):
        """R parity: stats::runif(10) is re-evaluated each call."""
        np.random.seed(42)
        g1 = points_grob()
        g2 = points_grob()
        assert not np.allclose(g1.x.values, g2.x.values)
        assert not np.allclose(g1.y.values, g2.y.values)

    def test_seed_reproducibility(self):
        """Same numpy seed → same default points across runs."""
        np.random.seed(123)
        g1 = points_grob()
        np.random.seed(123)
        g2 = points_grob()
        assert np.allclose(g1.x.values, g2.x.values)
        assert np.allclose(g1.y.values, g2.y.values)

    def test_default_units_native(self):
        """R: default.units = "native"."""
        g = points_grob()
        assert g.x._units[0] == "native"
        assert g.y._units[0] == "native"

    def test_explicit_coords_not_overridden(self):
        """When x/y are provided, defaults are not generated."""
        g = points_grob(x=[0.1, 0.2, 0.3], y=[0.4, 0.5, 0.6])
        assert len(g.x) == 3
        assert pytest.approx(list(g.x.values)) == [0.1, 0.2, 0.3]
