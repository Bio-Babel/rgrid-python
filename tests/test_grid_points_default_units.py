"""Regression test: ``grid_points`` ``default_units`` must match R.

R's ``grid.points`` and ``pointsGrob`` both default ``default.units =
"native"`` (primitives.R). Earlier ``grid_points`` defaulted to
``"npc"`` while ``points_grob`` used ``"native"`` — the wrapper / grob
mismatch would silently misinterpret bare numerics passed to the
convenience function.
"""

from __future__ import annotations

import inspect

from grid_py import grid_points, points_grob


class TestGridPointsDefaultUnits:
    def test_grid_points_default_units_is_native(self):
        sig = inspect.signature(grid_points)
        assert sig.parameters["default_units"].default == "native"

    def test_grid_points_matches_points_grob(self):
        """Wrapper and constructor must agree (sibling-consistency)."""
        g_sig = inspect.signature(grid_points)
        c_sig = inspect.signature(points_grob)
        assert (
            g_sig.parameters["default_units"].default
            == c_sig.parameters["default_units"].default
        )
