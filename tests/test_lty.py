"""Regression tests for grid_py._lty — R linetype resolution.

These tests pin the R-gold-standard behaviour established in the B7
audit (see r2py-toolkit/b7_repros_tmp/plan.md).  They cover three
classes of risk:

1. **Value-level R parity** — hex parsing, named lookups, lwd scaling
   (factor 1.0, verified by R cairo PNG measurement).
2. **R quirks** — string "0" rejected while integer 0 is BLANK, hex
   length must be in {2,4,6,8} (R: ``invalid line type: must be length
   2, 4, 6 or 8``).
3. **Renderer integration** — _apply_stroke routes blank, named, hex
   and integer lty through one resolver; path-collection (stroke_grob /
   fill_stroke_grob) inherits the same dash via singleton.  Guards
   against future regressions on commits ef7aee5 / b854b97 / 5488946 /
   8a11ef7 (see plan.md §7.2).
"""

from __future__ import annotations

import numpy as np
import pytest

from grid_py._lty import is_blank_lty, resolve_lty, valid_named_lty


# ---------------------------------------------------------------------------
# resolve_lty — value-level R parity
# ---------------------------------------------------------------------------


class TestResolveDashNamed:
    """Each R named lty must produce its R-gold dash array at lwd=1."""

    @pytest.mark.parametrize("name,expected", [
        ("dashed",   [4.0, 4.0]),            # LTY_DASHED = 0x44
        ("dotted",   [1.0, 3.0]),            # LTY_DOTTED = 0x31 → "13"
        ("dotdash",  [1.0, 3.0, 4.0, 3.0]),  # LTY_DOTDASH = 0x3431 → "1343"
        ("longdash", [7.0, 3.0]),            # LTY_LONGDASH = 0x37 → "73"
        ("twodash",  [2.0, 2.0, 6.0, 2.0]),  # LTY_TWODASH = 0x2622 → "2262"
    ])
    def test_named_dash_lwd1(self, name, expected):
        assert resolve_lty(name, lwd=1.0) == expected

    def test_solid_returns_None(self):
        assert resolve_lty("solid") is None

    def test_None_returns_None(self):
        assert resolve_lty(None) is None


class TestResolveLwdScaling:
    """Dash = nibble × lwd (factor 1.0, verified against R cairo PNG)."""

    @pytest.mark.parametrize("lwd,expected", [
        (1.0, [4.0, 4.0]),
        (1.5, [6.0, 6.0]),
        (2.0, [8.0, 8.0]),
        (4.0, [16.0, 16.0]),
        (0.5, [2.0, 2.0]),
    ])
    def test_dashed_scales_linearly(self, lwd, expected):
        assert resolve_lty("dashed", lwd=lwd) == expected

    def test_named_equals_hex_byte_identical(self):
        """The named lty 'dashed' and the hex '44' must produce identical output."""
        for lwd in (0.5, 1.0, 1.5, 2.0, 3.7):
            assert resolve_lty("dashed", lwd=lwd) == resolve_lty("44", lwd=lwd)


class TestResolveHexLong:
    """Long hex (8-char) patterns from pal_linetype must expand correctly."""

    def test_F4448444(self):
        # nibbles low-to-high: F, 4, 4, 4, 8, 4, 4, 4
        assert resolve_lty("F4448444", lwd=1.0) == \
            [15.0, 4.0, 4.0, 4.0, 8.0, 4.0, 4.0, 4.0]

    def test_12223242(self):
        assert resolve_lty("12223242", lwd=1.0) == \
            [1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 4.0, 2.0]

    def test_224282F2(self):
        assert resolve_lty("224282F2", lwd=1.0) == \
            [2.0, 2.0, 4.0, 2.0, 8.0, 2.0, 15.0, 2.0]

    def test_F1(self):
        assert resolve_lty("F1", lwd=1.0) == [15.0, 1.0]


class TestResolveZeroNibbleTerminator:
    """R rule (cairoFns.c::CairoLineType): nibble 0 terminates expansion.

    R rejects odd-length hex up front so the only place a zero nibble
    appears is at the end of a longer valid pattern like '4400'.
    """

    def test_4400_truncates(self):
        # "4400" → low-high nibbles [4, 4, 0, 0] → stop at first 0
        assert resolve_lty("4400", lwd=1.0) == [4.0, 4.0]

    def test_FF00_truncates(self):
        assert resolve_lty("FF00", lwd=1.0) == [15.0, 15.0]


class TestResolveIntegerForms:
    """R par convention: integer 1..6 maps to the 6 named ltys.

    Integer 0 is BLANK and must be caught by is_blank_lty *before*
    resolve_lty (calling resolve_lty(0) raises to surface the omission).
    """

    @pytest.mark.parametrize("i,expected_name", [
        (1, "solid"), (2, "dashed"), (3, "dotted"),
        (4, "dotdash"), (5, "longdash"), (6, "twodash"),
    ])
    def test_int_matches_named(self, i, expected_name):
        assert resolve_lty(i, lwd=1.0) == resolve_lty(expected_name, lwd=1.0)

    def test_zero_raises_to_warn_caller(self):
        with pytest.raises(ValueError, match="is BLANK"):
            resolve_lty(0)

    def test_seven_raises(self):
        with pytest.raises(ValueError, match="1..6"):
            resolve_lty(7)

    def test_numpy_integer_accepted(self):
        assert resolve_lty(np.int64(2)) == [4.0, 4.0]


class TestResolveListWrappedInput:
    """Per R recycling, lty may arrive as a single-element list/array."""

    def test_list_wrapped(self):
        assert resolve_lty(["dashed"]) == resolve_lty("dashed")

    def test_tuple_wrapped(self):
        assert resolve_lty(("44",), lwd=2.0) == resolve_lty("44", lwd=2.0)

    def test_numpy_array_wrapped(self):
        arr = np.array(["1343"], dtype=object)
        assert resolve_lty(arr, lwd=1.0) == [1.0, 3.0, 4.0, 3.0]


# ---------------------------------------------------------------------------
# resolve_lty — error paths (R-gold; principle 4: fail loud)
# ---------------------------------------------------------------------------


class TestResolveErrors:

    def test_invalid_hex_chars(self):
        with pytest.raises(ValueError, match="invalid hex digit"):
            resolve_lty("xy")

    @pytest.mark.parametrize("bad", ["4", "444", "44444", "4444444",
                                      "444444444", "F4448444F"])
    def test_invalid_hex_length(self, bad):
        # R: invalid line type: must be length 2, 4, 6 or 8
        with pytest.raises(ValueError, match=r"length \d+ not in"):
            resolve_lty(bad)

    def test_str_zero_is_error_not_blank(self):
        """R quirk: int 0 is BLANK, but str '0' fails length check."""
        with pytest.raises(ValueError, match="length 1 not in"):
            resolve_lty("0")

    def test_str_zero_not_blank(self):
        assert not is_blank_lty("0")     # confirms the asymmetry

    def test_NA_sentinel_to_resolve_lty_raises(self):
        """Caller must check is_blank_lty([None]) before resolve_lty."""
        with pytest.raises(ValueError, match="must be handled by caller"):
            resolve_lty([None])


# ---------------------------------------------------------------------------
# is_blank_lty
# ---------------------------------------------------------------------------


class TestIsBlankLty:

    def test_int_zero_is_blank(self):
        assert is_blank_lty(0)

    def test_int_nonzero_not_blank(self):
        for i in [1, 2, 6, -1, 7]:
            assert not is_blank_lty(i)

    def test_blank_string(self):
        assert is_blank_lty("blank")
        assert is_blank_lty("BLANK")
        assert is_blank_lty("Blank")

    def test_NA_sentinel(self):
        """Gpar(lty=None) is stored as [None] and must be blank."""
        assert is_blank_lty([None])

    def test_scalar_None_is_NOT_blank(self):
        """Bare None means 'unspecified' → falls back to solid, not blank."""
        assert not is_blank_lty(None)

    def test_named_solid_not_blank(self):
        assert not is_blank_lty("solid")
        assert not is_blank_lty("dashed")

    def test_str_zero_not_blank(self):
        """R asymmetry: int 0 is blank, str '0' is not."""
        assert not is_blank_lty("0")


# ---------------------------------------------------------------------------
# valid_named_lty
# ---------------------------------------------------------------------------


class TestValidNamedLty:

    def test_contains_all_seven(self):
        v = valid_named_lty()
        assert v == frozenset({"solid", "dashed", "dotted", "dotdash",
                               "longdash", "twodash", "blank"})


# ---------------------------------------------------------------------------
# Renderer integration — guards against red-line regressions
# (see b7_repros_tmp/plan.md §7.2)
# ---------------------------------------------------------------------------


class TestRendererIntegration:
    """End-to-end: render a dashed line via CairoRenderer and verify it dashes."""

    def _render_line(self, lty, lwd=2.0, w=4.0, h=0.4, dpi=100):
        import grid_py
        from grid_py._draw import grid_newpage, grid_draw
        grid_newpage(width=w, height=h, dpi=dpi, bg="white")
        g = grid_py.lines_grob(
            x=np.array([0.05, 0.95]), y=np.array([0.5, 0.5]),
            default_units="npc",
            gp=grid_py.Gpar(lty=lty, lwd=lwd, col="black"),
        )
        grid_draw(g)
        return grid_py.get_state().get_renderer().to_png_bytes()

    def _count_dashes_in_middle_row(self, png_bytes):
        """Return number of contiguous black runs in the image's central row."""
        from io import BytesIO
        from PIL import Image
        img = np.array(Image.open(BytesIO(png_bytes)).convert("L"))
        row = img[img.shape[0] // 2]
        is_black = row < 128
        runs = 0
        in_run = False
        for b in is_black:
            if b and not in_run:
                runs += 1
                in_run = True
            elif not b:
                in_run = False
        return runs

    def test_solid_produces_one_run(self):
        png = self._render_line("solid", lwd=2)
        assert self._count_dashes_in_middle_row(png) == 1

    def test_named_dashed_produces_many_runs(self):
        png = self._render_line("dashed", lwd=2)
        # An ~400px line dashed [4,4] @ lwd=2 → set_dash=[8,8] → ~25 dashes
        n = self._count_dashes_in_middle_row(png)
        assert n > 10, f"expected many dashes, got {n}"

    def test_hex_44_equals_named_dashed(self):
        """Cairo path: lty='44' must produce same PNG as lty='dashed'."""
        png_hex = self._render_line("44", lwd=2)
        png_named = self._render_line("dashed", lwd=2)
        # ImageMagick compare would be ideal; minimum-viable: same dash count
        assert self._count_dashes_in_middle_row(png_hex) == \
            self._count_dashes_in_middle_row(png_named)

    def test_hex_long_renders_without_silent_solid(self):
        """B7 root cause: hex strings used to silently render solid."""
        png = self._render_line("F4448444", lwd=2)
        n = self._count_dashes_in_middle_row(png)
        assert n > 1, "hex lty must dash (was silently solid before B7 fix)"

    def test_blank_lty_skips_stroke(self):
        """lty='blank' / lty=0 must produce zero stroke pixels (LTY_BLANK)."""
        from io import BytesIO
        from PIL import Image
        for lty_val in ("blank", 0):
            png = self._render_line(lty_val, lwd=4)
            img = np.array(Image.open(BytesIO(png)).convert("L"))
            black = int((img < 128).sum())
            assert black == 0, f"lty={lty_val!r} expected 0 black px, got {black}"

    def test_lwd_scales_dash_period(self):
        """commit ef7aee5: dash and line width share user-space.
        Doubling lwd must double the dash period."""
        png1 = self._render_line("44", lwd=1)
        png2 = self._render_line("44", lwd=2)
        n1 = self._count_dashes_in_middle_row(png1)
        n2 = self._count_dashes_in_middle_row(png2)
        # lwd=2 dash period is 2x → roughly half as many dashes in the same span
        # tolerate ±25 % for anti-aliasing edge effects
        assert n1 > 0 and n2 > 0
        ratio = n1 / n2
        assert 1.5 < ratio < 2.5, \
            f"expected n(lwd=1)/n(lwd=2) ~= 2, got {n1}/{n2} = {ratio:.2f}"

    def test_invalid_lty_raises_not_silent_solid(self):
        """B7 acceptance: unknown lty must raise, not silently solid."""
        import grid_py
        with pytest.raises(ValueError):
            grid_py.Gpar(lty="zigzag")


class TestPathCollectionInheritance:
    """B7 plan §7.2 B): stroke_grob / fill_stroke_grob use _apply_stroke
    via end_path_stroke, so our lty fix auto-propagates to those code
    paths.  This test pins that behaviour to prevent future refactors of
    the path-collection finalizers from silently regressing."""

    def test_stroke_grob_dash_propagates(self):
        import grid_py
        from grid_py._draw import grid_newpage, grid_draw
        from io import BytesIO
        from PIL import Image

        grid_newpage(width=4, height=0.4, dpi=100, bg="white")
        # Build a stroke grob over a lines grob with hex lty.
        inner = grid_py.lines_grob(
            x=np.array([0.05, 0.95]), y=np.array([0.5, 0.5]),
            default_units="npc",
        )
        outer = grid_py.stroke_grob(
            inner, gp=grid_py.Gpar(lty="44", lwd=2, col="black"),
        )
        grid_draw(outer)
        png = grid_py.get_state().get_renderer().to_png_bytes()
        img = np.array(Image.open(BytesIO(png)).convert("L"))
        row = img[img.shape[0] // 2]
        is_black = row < 128
        runs = sum(1 for i in range(1, len(is_black))
                   if is_black[i] and not is_black[i - 1])
        assert runs > 1, "stroke_grob with hex lty must dash via shared _apply_stroke"


class TestWebRendererSerialisation:
    """B7 §4.3: lty is pre-resolved to gpar.dash (no string lty in JSON).

    Tests target the serialiser function directly to avoid coupling to
    the higher-level grid_newpage / get_state plumbing.
    """

    def _serialise(self, gp):
        """Call _serialise_gpar with minimal stub defs/id_gen."""
        from grid_py.renderer_web import _serialise_gpar
        from grid_py._scene_graph import DefsCollection, _IdGenerator
        return _serialise_gpar(gp, DefsCollection(), _IdGenerator())

    def test_named_dashed_serialised_as_dash_array(self):
        import grid_py
        result = self._serialise(grid_py.Gpar(lty="dashed", lwd=2.0))
        assert result.get("dash") == [8.0, 8.0]
        assert "lty" not in result, f"raw lty string still in payload: {result}"

    def test_hex_lty_serialised_as_dash_array(self):
        import grid_py
        result = self._serialise(grid_py.Gpar(lty="44", lwd=2.0))
        assert result.get("dash") == [8.0, 8.0]
        assert "lty" not in result

    def test_solid_no_dash_field(self):
        import grid_py
        result = self._serialise(grid_py.Gpar(lty="solid", lwd=2.0))
        # solid -> resolve_lty returns None -> no "dash" key written
        assert "dash" not in result
        assert "lty" not in result

    def test_blank_lty_sets_skip_stroke(self):
        import grid_py
        for lty_val in ("blank", 0):
            result = self._serialise(grid_py.Gpar(lty=lty_val, lwd=2.0))
            assert result.get("skip_stroke") is True, \
                f"lty={lty_val!r} should set skip_stroke=True, got {result}"
            assert "dash" not in result
            assert "lty" not in result

    def test_lwd_scales_dash(self):
        """Web side uses raw lwd (no _lwd_to_user conversion)."""
        import grid_py
        r1 = self._serialise(grid_py.Gpar(lty="dashed", lwd=1.0))
        r2 = self._serialise(grid_py.Gpar(lty="dashed", lwd=2.0))
        r4 = self._serialise(grid_py.Gpar(lty="dashed", lwd=4.0))
        assert r1["dash"] == [4.0, 4.0]
        assert r2["dash"] == [8.0, 8.0]
        assert r4["dash"] == [16.0, 16.0]
