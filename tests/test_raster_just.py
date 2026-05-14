"""Regression tests for ``raster_grob`` justification against R semantics.

Gold standard: R ``grid::justifyX(x, w, hjust) = x - w*hjust`` and
``grid::justifyY(y, h, vjust) = y - h*vjust`` (``grid/src/just.c`` lines 30-53).
R operates in y-up NPC; ``grid_py.resolve_y`` already returns y-down Cairo
device pixels, so the renderer dispatcher must flip the vjust term —
``y - h*(1-vjust)`` — matching what ``draw_rect`` does
(``renderer.py:815``).

Visual fixture: a 0.5x0.5 npc red tile anchored at viewport centre. For each
``just=`` value we pixel-probe the four corner quadrants of a 4x4 inch PNG and
compare to a separately verified R ``grid::grid.raster`` rendering.
"""

from __future__ import annotations

import numpy as np
import pytest

PIL = pytest.importorskip("PIL.Image")

from grid_py import CairoRenderer, Unit, grid_draw, raster_grob
from grid_py._state import get_state


_RED = np.tile(np.array([255, 0, 0, 255], dtype=np.uint8), (2, 2, 1))

# Expected pixel labels at the four corner quadrants for each justification.
# 'R' = red (tile occupies that quadrant); 'W' = white (background).
# These were cross-validated against R 4.5 ``grid::grid.raster`` output.
_EXPECTED = {
    "centre":             {"TL": "W", "TR": "W", "BL": "W", "BR": "W"},
    ("left", "top"):      {"TL": "W", "TR": "W", "BL": "W", "BR": "R"},
    ("right", "top"):     {"TL": "W", "TR": "W", "BL": "R", "BR": "W"},
    ("left", "bottom"):   {"TL": "W", "TR": "R", "BL": "W", "BR": "W"},
    ("right", "bottom"):  {"TL": "R", "TR": "W", "BL": "W", "BR": "W"},
}

_LABEL = {(255, 0, 0): "R", (255, 255, 255): "W"}


def _render_and_probe(just, tmp_path):
    fn = tmp_path / f"raster_just_{str(just).replace(' ', '')}.png"
    r = CairoRenderer(width=4, height=4, dpi=100,
                      surface_type="image", bg="white")
    get_state()._renderer = r
    grid_draw(raster_grob(
        _RED,
        x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
        width=Unit(0.5, "npc"), height=Unit(0.5, "npc"),
        just=just, interpolate=False,
    ))
    r.write_to_png(str(fn))
    im = PIL.open(str(fn)).convert("RGB")
    W, H = im.size
    samples = {
        "TL": im.getpixel((int(W * 0.15), int(H * 0.15))),
        "TR": im.getpixel((int(W * 0.85), int(H * 0.15))),
        "BL": im.getpixel((int(W * 0.15), int(H * 0.85))),
        "BR": im.getpixel((int(W * 0.85), int(H * 0.85))),
    }
    return {q: _LABEL.get(px, "?") for q, px in samples.items()}


@pytest.mark.parametrize("just", list(_EXPECTED.keys()))
def test_raster_grob_justification_matches_r(just, tmp_path):
    """R `grid.raster(just=just)` placement matches `raster_grob(just=just)`.

    This test would fail before the y-flip fix in ``_draw.py`` for every
    ``just`` whose vjust != 0.5 — the rasters landed exactly one viewport-height
    away from their R counterparts.
    """
    got = _render_and_probe(just, tmp_path)
    assert got == _EXPECTED[just], (
        f"just={just!r}: got {got}, expected (R-verified) {_EXPECTED[just]}"
    )
