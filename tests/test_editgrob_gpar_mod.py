"""Regression tests for the ``editGrob`` gp merge semantics.

R's ``editGrob`` calls ``mod.gpar`` (gpar.R:298-304), which does a
**plain key overwrite** — unmentioned keys are kept, no cumulative
multiplication of ``cex`` / ``alpha`` / ``lex``. This is distinct from
``set.gpar`` (which viewport-push uses) where those three params are
multiplied across the inheritance chain.

Earlier ``_grob.py`` mis-routed editGrob through ``Gpar.merge`` (which
didn't exist, so the ``hasattr`` fallback silently replaced the entire
gp), losing the original gp's other fields. Now editGrob uses
``Gpar._mod`` (mod.gpar semantics) and viewport push keeps using
``Gpar._merge`` (set.gpar semantics).
"""

from __future__ import annotations

import pytest

from grid_py import (
    Gpar,
    Viewport,
    edit_grob,
    get_gpar,
    grid_newpage,
    push_viewport,
    rect_grob,
)


class TestEditGrobUsesModGpar:
    """``editGrob(g, gp=...)`` must preserve original gp's other fields."""

    def test_other_fields_preserved(self):
        """R: editGrob(rectGrob(gp=gpar(col='red', lwd=2)), gp=gpar(col='blue'))
        yields gp={col='blue', lwd=2} — lwd survives."""
        g = rect_grob(gp=Gpar(col="red", lwd=2))
        g2 = edit_grob(g, gp=Gpar(col="blue"))
        assert g2.gp.get("col") == "blue"
        assert g2.gp.get("lwd") == 2.0

    def test_cumulative_params_replaced_not_multiplied(self):
        """R mod.gpar overwrites cex (3, not 2*3=6) — distinct from set.gpar."""
        g = rect_grob(gp=Gpar(col="red", cex=2))
        g2 = edit_grob(g, gp=Gpar(col="blue", cex=3))
        assert g2.gp.get("cex") == 3.0  # NOT 6 (would be cumulative)
        assert g2.gp.get("col") == "blue"

    def test_gp_none_clears(self):
        """R: editGrob(..., gp=NULL) clears the gp entirely."""
        g = rect_grob(gp=Gpar(col="red"))
        g2 = edit_grob(g, gp=None)
        assert g2.gp is None

    def test_no_prior_gp_just_sets(self):
        """If grob has no prior gp, edit just installs the new gp."""
        g = rect_grob()  # gp is None
        g2 = edit_grob(g, gp=Gpar(col="green"))
        assert g2.gp.get("col") == "green"


class TestViewportPushStillCumulative:
    """``pushViewport`` must keep using set.gpar (cumulative) semantics."""

    def test_nested_cex_multiplies(self):
        """R: nested pushViewport(gp=gpar(cex=2)) then gpar(cex=3) → cex=6."""
        grid_newpage()
        push_viewport(Viewport(gp=Gpar(cex=2)))
        push_viewport(Viewport(gp=Gpar(cex=3)))
        cex = get_gpar(["cex"]).get("cex")
        assert cex == 6.0  # 2 * 3

    def test_nested_alpha_multiplies(self):
        """R: nested pushViewport(gp=gpar(alpha=0.5)) twice → alpha=0.25."""
        grid_newpage()
        push_viewport(Viewport(gp=Gpar(alpha=0.5)))
        push_viewport(Viewport(gp=Gpar(alpha=0.5)))
        alpha = get_gpar(["alpha"]).get("alpha")
        assert alpha == pytest.approx(0.25)


class TestGparModInternals:
    """Direct unit tests on ``Gpar._mod``."""

    def test_mod_is_plain_overwrite(self):
        """`_mod` overwrites keys; doesn't multiply cumulative params."""
        a = Gpar(col="red", cex=2, lwd=3)
        b = Gpar(col="blue", cex=4)
        merged = a._mod(b)
        assert merged.get("col") == "blue"
        assert merged.get("cex") == 4.0  # NOT 8 (would be cumulative)
        assert merged.get("lwd") == 3.0  # kept from a

    def test_merge_is_cumulative(self):
        """`_merge` preserves the existing cumulative behaviour for set.gpar."""
        a = Gpar(col="red", cex=2, lwd=3)
        b = Gpar(col="blue", cex=4)
        merged = a._merge(b)
        # a is child, b is parent — child's cex multiplied by parent's cex
        assert merged.get("col") == "red"  # child wins
        assert merged.get("cex") == 8.0  # 2 * 4 cumulative
