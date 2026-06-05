"""Tests for character / mixed ``pch`` support in points grobs.

R gold standard (``grid:::valid.pch``, grid 4.5.3):

* zero-length ``pch`` → error
* ``None`` → ``1`` (written R source; ``length(NULL)==0`` also errors in R,
  but ``None`` is Python's natural "use default" sentinel)
* character ``pch`` kept as character (``"."`` stays ``"."``)
* numeric ``pch`` coerced to ``int`` (floats truncated, like ``as.integer``)
* mixed ``[".", 19, "A"]`` → object array ``['.', '19', 'A']`` (R's ``c()``
  coerces to a character vector)

Engine semantics (verified by rendering in grid 4.5.3):

* numeric pch → symbols 0-25
* ``pch="."`` → a *tiny* dot (size ~1px at default cex, independent of the
  symbol size; grows only with large cex)
* any other character → that glyph, drawn centred (only the *first*
  character of a multi-char string is used, matching R)
"""

from __future__ import annotations

import numpy as np
import pytest

from grid_py._primitives import valid_pch, points_grob
from grid_py.renderer import CairoRenderer
from grid_py._gpar import Gpar
from grid_py._state import get_state
from grid_py._draw import grid_newpage, grid_draw


@pytest.fixture(autouse=True)
def _reset_state():
    state = get_state()
    state.reset()
    yield


# ------------------------------------------------------------------ #
# valid_pch — R parity                                               #
# ------------------------------------------------------------------ #

class TestValidPch:

    def test_dot_kept_as_character(self):
        assert valid_pch(".") == "."
        assert isinstance(valid_pch("."), str)

    def test_letter_kept_as_character(self):
        assert valid_pch("A") == "A"

    def test_numeric_int_passthrough(self):
        assert valid_pch(19) == 19
        assert isinstance(valid_pch(19), int)

    def test_numeric_float_truncated(self):
        # R's as.integer truncates toward zero
        assert valid_pch(19.7) == 19
        assert isinstance(valid_pch(19.7), int)

    def test_none_defaults_to_one(self):
        assert valid_pch(None) == 1

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="zero-length"):
            valid_pch([])
        with pytest.raises(ValueError, match="zero-length"):
            valid_pch(np.array([], dtype=int))

    def test_all_numeric_array_to_int(self):
        out = valid_pch([1, 2, 3])
        assert out.dtype == int
        assert list(out) == [1, 2, 3]

    def test_mixed_array_coerced_to_object_strings(self):
        # R: c(".", 19, "A") -> c(".", "19", "A")
        out = valid_pch([".", 19, "A"])
        assert out.dtype == object
        assert list(out) == [".", "19", "A"]

    def test_mixed_array_numeric_becomes_string(self):
        out = valid_pch([19, "."])
        assert list(out) == ["19", "."]


# ------------------------------------------------------------------ #
# points_grob applies valid_pch (R validDetails.points)             #
# ------------------------------------------------------------------ #

class TestPointsGrobValidation:

    def test_grob_keeps_character_pch(self):
        g = points_grob(x=[0.5], y=[0.5], pch=".")
        assert g.pch == "."

    def test_grob_coerces_numeric_pch(self):
        g = points_grob(x=[0.5], y=[0.5], pch=19.9)
        assert g.pch == 19

    def test_grob_mixed_pch(self):
        g = points_grob(x=[0.2, 0.5, 0.8], y=[0.5, 0.5, 0.5],
                        pch=[".", 19, "A"])
        assert list(g.pch) == [".", "19", "A"]


# ------------------------------------------------------------------ #
# Renderer — character pch does not crash and renders sensibly       #
# ------------------------------------------------------------------ #

class TestDrawPointsCharacter:

    @pytest.fixture
    def renderer(self):
        return CairoRenderer(width=4, height=2, dpi=100, bg="white")

    def test_dot_does_not_crash(self, renderer):
        x = np.array([100.0])
        y = np.array([100.0])
        renderer.draw_points(x, y, size=20.0, pch=".", gp=Gpar(col="black"))

    def test_letter_does_not_crash(self, renderer):
        x = np.array([100.0])
        y = np.array([100.0])
        renderer.draw_points(x, y, size=20.0, pch="A", gp=Gpar(col="black"))

    def test_numeric_still_works(self, renderer):
        x = np.array([100.0])
        y = np.array([100.0])
        renderer.draw_points(x, y, size=20.0, pch=19, gp=Gpar(col="black"))

    def test_mixed_array_does_not_crash(self, renderer):
        x = np.array([60.0, 160.0, 260.0, 360.0])
        y = np.full(4, 100.0)
        renderer.draw_points(
            x, y, size=20.0, pch=[".", 19, "A", "1"], gp=Gpar(col="black"),
        )

    def _ink_bbox(self, renderer):
        """Return (w, h) of the non-white ink bounding box in pixels."""
        import io
        from PIL import Image
        data = renderer.to_png_bytes()
        im = np.array(Image.open(io.BytesIO(data)).convert("L"))
        ys, xs = np.where(im < 200)
        if len(xs) == 0:
            return (0, 0)
        return (int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))

    def test_dot_is_tiny_relative_to_symbol(self):
        """R: pch='.' is ~1px while a symbol-19 disc is much larger."""
        pytest.importorskip("PIL")
        r_dot = CairoRenderer(width=2, height=2, dpi=72, bg="white")
        r_dot.draw_points(np.array([72.0]), np.array([72.0]),
                          size=20.0, pch=".", gp=Gpar(col="black"))
        dot_w, dot_h = self._ink_bbox(r_dot)

        r_sym = CairoRenderer(width=2, height=2, dpi=72, bg="white")
        r_sym.draw_points(np.array([72.0]), np.array([72.0]),
                          size=20.0, pch=19, gp=Gpar(col="black"))
        sym_w, sym_h = self._ink_bbox(r_sym)

        # Dot must have ink (not vanish) but be far smaller than symbol 19.
        assert dot_w >= 1 and dot_h >= 1
        assert dot_w < sym_w
        assert dot_h < sym_h

    def test_dot_size_independent_of_grob_size(self):
        """R: pch='.' size does not scale with the points-grob size."""
        pytest.importorskip("PIL")
        sizes = []
        for sz in (10.0, 30.0, 60.0):
            r = CairoRenderer(width=2, height=2, dpi=72, bg="white")
            r.draw_points(np.array([72.0]), np.array([72.0]),
                          size=sz, pch=".", gp=Gpar(col="black"))
            sizes.append(self._ink_bbox(r))
        # All identical regardless of size argument.
        assert sizes[0] == sizes[1] == sizes[2]

    def test_letter_renders_ink(self):
        """A glyph pch leaves visible ink (the letter)."""
        pytest.importorskip("PIL")
        r = CairoRenderer(width=2, height=2, dpi=100, bg="white")
        r.draw_points(np.array([100.0]), np.array([100.0]),
                      size=20.0, pch="A", gp=Gpar(col="black"))
        w, h = self._ink_bbox(r)
        assert w >= 2 and h >= 2


# ------------------------------------------------------------------ #
# End-to-end via grid_draw (points grob through the draw path)       #
# ------------------------------------------------------------------ #

class TestEndToEndPointsGrob:

    def test_dot_grob_draws(self):
        grid_newpage()
        g = points_grob(x=[0.2, 0.5, 0.8], y=[0.5, 0.5, 0.5], pch=".")
        grid_draw(g)  # must not raise

    def test_mixed_grob_draws(self):
        grid_newpage()
        g = points_grob(x=[0.2, 0.5, 0.8], y=[0.5, 0.5, 0.5],
                        pch=[".", 19, "A"])
        grid_draw(g)  # must not raise


# ------------------------------------------------------------------ #
# draw_raster fail-loud on malformed (1-D) colour-string input       #
# ------------------------------------------------------------------ #

class TestDrawRasterFailLoud:

    def test_one_d_string_raster_raises(self):
        r = CairoRenderer(width=2, height=2, dpi=72, bg="white")
        with pytest.raises(ValueError, match="must be 2-D"):
            r.draw_raster(np.array(["red", "green", "blue"]),
                          0.0, 0.0, 10.0, 10.0)

    def test_two_d_string_raster_works(self):
        r = CairoRenderer(width=2, height=2, dpi=72, bg="white")
        # 2-D colour matrix must still render without error.
        r.draw_raster(np.array([["red", "green"], ["blue", "yellow"]]),
                      0.0, 0.0, 10.0, 10.0)

    def test_two_d_numeric_raster_works(self):
        r = CairoRenderer(width=2, height=2, dpi=72, bg="white")
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[:, :, 0] = 255
        r.draw_raster(img, 0.0, 0.0, 10.0, 10.0)
