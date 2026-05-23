"""Regression: silent ``except Exception: pass`` swallows surface as warnings.

Per CLAUDE.md global rule 2: ``try/except`` should be sparing and
should not silently mask real errors. Earlier 14 sites in
``_draw.py`` / ``renderer.py`` / ``_patterns.py`` /
``_renderer_base.py`` used bare ``except Exception: pass``. They are
now narrowed to specific exception types, and the user-meaningful
ones (grob render, arrow length, grob unit eval, grob-coord resolve,
mask grob, tiling-pattern grob, group composition) emit a
``UserWarning`` so the user can tell their input was wrong instead of
seeing a silently degraded plot.
"""

from __future__ import annotations

import warnings

import pytest


class TestNoBareExceptExceptionLeft:
    """Static check: no bare ``except Exception`` should be reintroduced
    in the renderer / drawing hot paths."""

    def test_no_bare_except_in_target_files(self):
        from pathlib import Path
        import re
        root = Path(__file__).resolve().parent.parent / "grid_py"
        targets = ["_draw.py", "renderer.py", "_patterns.py", "_renderer_base.py"]
        pattern = re.compile(r"^\s*except\s+Exception\s*:\s*$", re.MULTILINE)
        offenders = []
        for fname in targets:
            fp = root / fname
            text = fp.read_text()
            for m in pattern.finditer(text):
                line_no = text[: m.start()].count("\n") + 1
                offenders.append(f"{fname}:{line_no}")
        assert not offenders, (
            "Bare `except Exception:` re-introduced (CLAUDE.md rule 2 — "
            "narrow types + emit UserWarning instead): " + ", ".join(offenders)
        )


class TestGrobUnitEvalFallbackWarns:
    """Buggy user-grob ``width_details`` should now surface a warning,
    not silently report 0 inches."""

    def test_buggy_width_details_emits_warning(self):
        from grid_py import grid_newpage, rect_grob
        from grid_py._renderer_base import GridRenderer
        # Build a renderer with a deliberately buggy grob whose
        # width_details raises an AttributeError when called.
        class BrokenGrob:
            x = None; y = None; vp = None; gp = None
            def width_details(self): raise AttributeError("intentional")
            def height_details(self): return None
            def pre_draw_details(self): pass
            def post_draw_details(self): pass

        grid_newpage()
        from grid_py._state import get_state
        renderer = get_state()._renderer
        assert renderer is not None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            # Trigger _grob_xy_inches_at_theta — exercises the
            # _details_inches fallback path
            renderer._grob_xy_inches_at_theta(BrokenGrob(), "grobx", 0.0)

        msgs = [str(w.message) for w in caught
                if issubclass(w.category, UserWarning)]
        assert any("width_details failed" in m for m in msgs), (
            f"Expected width_details warning; got: {msgs}"
        )
