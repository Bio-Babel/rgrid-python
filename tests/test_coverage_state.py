"""Tests to improve coverage of grid_py._state."""

import numpy as np
import pytest

from grid_py._gpar import Gpar
from grid_py._state import (
    GridState,
    get_state,
    _make_root_viewport,
    _vp_attr,
    _vp_set_attr,
    _vp_children,
    _vp_name,
    _vp_parent,
)


@pytest.fixture(autouse=True)
def _reset():
    state = get_state()
    state.reset()
    yield
    state.reset()


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_make_root_viewport(self):
        # ``_make_root_viewport`` now returns a ``SimpleNamespace`` instead
        # of a dict so that ``current_viewport().xscale`` works as the
        # R-style ``current.viewport()$xscale`` (attribute access).
        # The duck-typed ``_vp_attr`` helper still works either way.
        root = _make_root_viewport()
        assert root.name == "ROOT"
        assert root.parent is None
        assert isinstance(root.children, list)
        assert root.layout_pos is None
        # Default xscale/yscale before init_device binds a renderer.
        assert root.xscale == (0.0, 1.0)
        assert root.yscale == (0.0, 1.0)

    def test_vp_attr_dict(self):
        vp = {"name": "test", "x": 42}
        assert _vp_attr(vp, "name") == "test"
        assert _vp_attr(vp, "missing", "default") == "default"

    def test_vp_attr_object(self):
        class VP:
            name = "obj_vp"
        assert _vp_attr(VP(), "name") == "obj_vp"
        assert _vp_attr(VP(), "missing", "def") == "def"

    def test_vp_set_attr_dict(self):
        vp = {}
        _vp_set_attr(vp, "name", "new")
        assert vp["name"] == "new"

    def test_vp_set_attr_object(self):
        class VP:
            pass
        vp = VP()
        _vp_set_attr(vp, "name", "new")
        assert vp.name == "new"

    def test_vp_children(self):
        vp = {"children": [1, 2]}
        assert _vp_children(vp) == [1, 2]

    def test_vp_name(self):
        vp = {"name": "test"}
        assert _vp_name(vp) == "test"

    def test_vp_parent(self):
        parent = {"name": "parent"}
        child = {"name": "child", "parent": parent}
        assert _vp_parent(child) is parent
        assert _vp_parent(parent) is None


class TestGridStateSingleton:
    """Tests for GridState singleton pattern."""

    def test_singleton(self):
        s1 = GridState()
        s2 = GridState()
        assert s1 is s2

    def test_get_state(self):
        state = get_state()
        assert isinstance(state, GridState)

    def test_current_viewport_root(self):
        state = get_state()
        vp = state.current_viewport()
        assert _vp_name(vp) == "ROOT"


class TestViewportTree:
    """Tests for viewport tree manipulation."""

    def test_push_viewport(self):
        state = get_state()
        child = {"name": "child1", "parent": None, "children": []}
        state.push_viewport(child)
        assert _vp_name(state.current_viewport()) == "child1"

    def test_pop_viewport(self):
        state = get_state()
        child = {"name": "child1", "parent": None, "children": []}
        state.push_viewport(child)
        state.pop_viewport(1)
        assert _vp_name(state.current_viewport()) == "ROOT"

    def test_pop_to_root(self):
        state = get_state()
        c1 = {"name": "c1", "parent": None, "children": []}
        c2 = {"name": "c2", "parent": None, "children": []}
        state.push_viewport(c1)
        state.push_viewport(c2)
        state.pop_viewport(0)  # pop to root
        assert _vp_name(state.current_viewport()) == "ROOT"

    def test_pop_negative_raises(self):
        state = get_state()
        with pytest.raises(ValueError, match="non-negative"):
            state.pop_viewport(-1)

    def test_pop_past_root_raises(self):
        state = get_state()
        with pytest.raises(ValueError, match="Cannot pop past"):
            state.pop_viewport(1)

    def test_up_viewport(self):
        state = get_state()
        child = {"name": "child1", "parent": None, "children": []}
        state.push_viewport(child)
        state.up_viewport(1)
        assert _vp_name(state.current_viewport()) == "ROOT"

    def test_up_to_root(self):
        state = get_state()
        c1 = {"name": "c1", "parent": None, "children": []}
        c2 = {"name": "c2", "parent": None, "children": []}
        state.push_viewport(c1)
        state.push_viewport(c2)
        state.up_viewport(0)
        assert _vp_name(state.current_viewport()) == "ROOT"

    def test_up_negative_raises(self):
        state = get_state()
        with pytest.raises(ValueError, match="non-negative"):
            state.up_viewport(-1)

    def test_up_past_root_raises(self):
        state = get_state()
        with pytest.raises(ValueError, match="Cannot navigate above"):
            state.up_viewport(1)

    def test_down_viewport(self):
        state = get_state()
        child = {"name": "target", "parent": None, "children": []}
        state.push_viewport(child)
        state.up_viewport(1)  # back to root, child still in tree
        depth = state.down_viewport("target")
        assert depth == 1
        assert _vp_name(state.current_viewport()) == "target"

    def test_down_viewport_not_found(self):
        state = get_state()
        with pytest.raises(LookupError, match="not found"):
            state.down_viewport("nonexistent")

    def test_down_viewport_case_insensitive(self):
        state = get_state()
        child = {"name": "MyVP", "parent": None, "children": []}
        state.push_viewport(child)
        state.up_viewport(1)
        depth = state.down_viewport("myvp", strict=False)
        assert depth == 1

    def test_seek_viewport(self):
        state = get_state()
        c1 = {"name": "level1", "parent": None, "children": []}
        c2 = {"name": "level2", "parent": None, "children": []}
        state.push_viewport(c1)
        state.push_viewport(c2)
        state.up_viewport(0)  # back to root
        depth = state.seek_viewport("level2")
        assert depth == 2

    def test_seek_viewport_not_found(self):
        state = get_state()
        with pytest.raises(LookupError, match="not found"):
            state.seek_viewport("nonexistent")


class TestVpQueries:
    """Tests for viewport query methods."""

    def test_current_vp_path(self):
        state = get_state()
        path = state.current_vp_path()
        assert path == "ROOT"

    def test_current_vp_path_nested(self):
        state = get_state()
        c1 = {"name": "panel", "parent": None, "children": []}
        state.push_viewport(c1)
        path = state.current_vp_path()
        assert path == "ROOT/panel"

    def test_current_vp_tree(self):
        state = get_state()
        tree = state.current_vp_tree()
        assert _vp_name(tree) == "ROOT"

    def test_current_transform(self):
        state = get_state()
        t = state.current_transform()
        np.testing.assert_array_equal(t, np.eye(3))

    def test_current_rotation(self):
        state = get_state()
        r = state.current_rotation()
        assert r == 0.0

    def test_current_rotation_nested(self):
        state = get_state()
        c1 = {"name": "c1", "parent": None, "children": [], "rotation": 30.0}
        state.push_viewport(c1)
        assert state.current_rotation() == 30.0

    def test_current_parent_at_root(self):
        state = get_state()
        assert state.current_parent() is None

    def test_current_parent_nested(self):
        state = get_state()
        root = state.current_viewport()
        c1 = {"name": "c1", "parent": None, "children": []}
        state.push_viewport(c1)
        assert state.current_parent() is root


class TestGparManagement:
    """Tests for gpar stack."""

    def test_get_gpar_default(self):
        state = get_state()
        gp = state.get_gpar()
        assert isinstance(gp, Gpar)

    def test_set_gpar(self):
        state = get_state()
        gp = Gpar(col="red")
        state.set_gpar(gp)
        assert state.get_gpar() is gp


class TestDisplayList:
    """Tests for display list management."""

    def test_record_on(self):
        state = get_state()
        state.record("op1")
        assert "op1" in state.get_display_list()

    def test_record_off(self):
        state = get_state()
        state.set_display_list_on(False)
        state.record("op1")
        assert len(state.get_display_list()) == 0

    def test_get_display_list(self):
        state = get_state()
        dl = state.get_display_list()
        # DisplayList is a list-like container
        assert hasattr(dl, '__len__') and hasattr(dl, '__iter__')


class TestDeviceBinding:
    """Tests for device/renderer binding."""

    def test_init_device(self):
        state = get_state()
        state.init_device("fake_renderer", width_cm=20.0, height_cm=15.0)
        assert state.get_renderer() == "fake_renderer"

    def test_get_device(self):
        state = get_state()
        state.init_device("r")
        fig, ax = state.get_device()
        assert fig == "r"
        assert ax == "r"

    def test_no_renderer_initially(self):
        state = get_state()
        assert state.get_renderer() is None
