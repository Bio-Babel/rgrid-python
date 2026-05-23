"""R-parity tests for ``viewport(clip = grob)`` — R 4.1+ feature.

R's ``viewport`` accepts a grob (or ``GridPath`` / ``GridClipPath``)
as the ``clip`` argument and uses its geometry as the clipping
region (viewport.R:86-97 → ``createClipPath(as.path(grob))``). The
rgrid-python port:

1. ``_valid_clip`` coerces grob / GridPath / GridClipPath → GridClipPath.
2. ``_renderer_base._do_apply_clip_vtr`` dispatches on the value's
   type; for GridClipPath it calls the backend's ``_apply_clip_grob``.
3. The Cairo backend's ``_apply_clip_grob`` re-uses the existing
   ``_path_collecting`` mode (originally for R 4.2+ ``fillStroke``
   grobs): primitives drop their stroke / fill calls but still emit
   Cairo path commands. After the grob's geometry is in the current
   path, ``ctx.clip()`` applies it. ``_restore_clip`` (paired with
   the implicit ``ctx.save()``) reverts on viewport pop, the same
   pop machinery rect clips use.
"""

from __future__ import annotations

import math

import pytest


class TestValidateClipAcceptsGrob:
    def test_grob_coerced_to_gridclippath(self):
        from grid_py import Viewport, circle_grob, GridClipPath
        vp = Viewport(clip=circle_grob())
        assert isinstance(vp.clip, GridClipPath)

    def test_gridclippath_identity_preserved(self):
        from grid_py import Viewport, rect_grob, as_clip_path
        ccp = as_clip_path(rect_grob())
        vp = Viewport(clip=ccp)
        assert vp.clip is ccp

    def test_gridpath_coerced(self):
        # GridPath path-objects also valid per R 4.1+
        from grid_py import Viewport, GridClipPath
        from grid_py._path import GridPath
        gp = GridPath("placeholder")
        vp = Viewport(clip=gp)
        assert isinstance(vp.clip, GridClipPath)

    def test_string_sentinels_unchanged(self):
        from grid_py import Viewport
        assert Viewport(clip="on").clip is True
        assert Viewport(clip="off").clip is None
        assert Viewport(clip="inherit").clip is False

    def test_bool_unchanged(self):
        from grid_py import Viewport
        assert Viewport(clip=True).clip is True
        assert Viewport(clip=False).clip is False

    def test_invalid_value_raises(self):
        from grid_py import Viewport
        with pytest.raises(ValueError, match="invalid 'clip'"):
            Viewport(clip=42)


class TestRenderUnderGrobClip:
    def test_circle_clip_pixel_count_matches_R(self):
        """R/Py cross-validation: a red rect rendered inside a viewport
        clipped to a circle should produce ~pi*r^2 red pixels in both
        engines. Tolerance 2% (Cairo vs R's libcairo can differ by 1-2
        boundary pixels)."""
        import matplotlib
        matplotlib.use("Agg")
        from grid_py import (
            Viewport, circle_grob, rect_grob, Unit, Gpar,
            grid_newpage, push_viewport, pop_viewport, grid_draw,
        )
        from grid_py._state import get_state

        side = 200
        grid_newpage(width=side / 72, height=side / 72, dpi=72)
        grid_draw(rect_grob(gp=Gpar(fill="white", col=None)))
        push_viewport(Viewport(clip=circle_grob(
            x=Unit(0.5, "npc"), y=Unit(0.5, "npc"), r=Unit(0.3, "npc"),
        )))
        grid_draw(rect_grob(gp=Gpar(fill="red", col=None)))
        pop_viewport()

        import tempfile, os
        tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tf.close()
        try:
            get_state()._renderer.write_to_png(tf.name)
            try:
                from PIL import Image
                import numpy as np
            except ImportError:
                pytest.skip("PIL/numpy required for pixel-count test")
            arr = np.array(Image.open(tf.name).convert("RGB"))
        finally:
            os.unlink(tf.name)

        red_count = ((arr[..., 0] > 200) & (arr[..., 1] < 100)
                     & (arr[..., 2] < 100)).sum()
        expected = math.pi * (0.3 * side) ** 2
        ratio = red_count / expected
        assert 0.98 <= ratio <= 1.02, (
            f"clip-grob red-pixel count {red_count} vs expected "
            f"{expected:.0f} (ratio {ratio:.3f}) — circle clip not "
            "applied correctly"
        )

    def test_pop_restores_full_canvas(self):
        """After popping a clip-grob viewport, subsequent drawing must
        cover the whole canvas — i.e. clip state was properly torn down."""
        import matplotlib
        matplotlib.use("Agg")
        from grid_py import (
            Viewport, circle_grob, rect_grob, Unit, Gpar,
            grid_newpage, push_viewport, pop_viewport, grid_draw,
        )
        from grid_py._state import get_state

        side = 100
        grid_newpage(width=side / 72, height=side / 72, dpi=72)
        # First push+pop a circle clip
        push_viewport(Viewport(clip=circle_grob(r=Unit(0.3, "npc"))))
        grid_draw(rect_grob(gp=Gpar(fill="red", col=None)))
        pop_viewport()
        # Now draw a green rect — should cover the whole canvas
        grid_draw(rect_grob(gp=Gpar(fill="green", col=None)))

        import tempfile, os
        tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tf.close()
        try:
            get_state()._renderer.write_to_png(tf.name)
            try:
                from PIL import Image
                import numpy as np
            except ImportError:
                pytest.skip("PIL/numpy required")
            arr = np.array(Image.open(tf.name).convert("RGB"))
        finally:
            os.unlink(tf.name)

        green_count = ((arr[..., 1] > 100) & (arr[..., 0] < 100)
                       & (arr[..., 2] < 100)).sum()
        total = arr.shape[0] * arr.shape[1]
        ratio = green_count / total
        assert ratio > 0.9, (
            f"post-pop green covers only {ratio:.2%} of canvas — "
            "clip state was not torn down on pop"
        )
