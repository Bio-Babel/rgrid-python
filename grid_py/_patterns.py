"""
Gradient and pattern fill system for grid_py.

Python port of R's ``grid`` package pattern infrastructure
(``grid/R/patterns.R``).  Provides linear gradients, radial gradients,
and tiling patterns that can be used as fill values in graphical
parameter (gpar) objects.

Classes
-------
LinearGradient
    A two-point linear colour gradient.
RadialGradient
    A two-circle radial colour gradient.
Pattern
    A tiling pattern based on an arbitrary grob.

Functions
---------
linear_gradient
    Factory for :class:`LinearGradient`.
radial_gradient
    Factory for :class:`RadialGradient`.
pattern
    Factory for :class:`Pattern`.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np

from ._units import Unit, is_unit

__all__: List[str] = [
    "LinearGradient",
    "RadialGradient",
    "Pattern",
    "ResolvedPattern",
    "linear_gradient",
    "radial_gradient",
    "pattern",
    "is_pattern",
    "is_pattern_list",
    "is_resolved_pattern",
    "resolve_fill",
    "resolve_pattern",
    "record_grob_for_pattern_resolution",
    "record_gtree_for_pattern_resolution",
]

# Valid *extend* modes, matching R's ``match.arg`` choices.
_VALID_EXTEND: tuple[str, ...] = ("pad", "repeat", "reflect", "none")


def _validate_extend(extend: str) -> str:
    """Return *extend* if valid, otherwise raise ``ValueError``.

    Parameters
    ----------
    extend : str
        One of ``"pad"``, ``"repeat"``, ``"reflect"``, or ``"none"``.

    Returns
    -------
    str
        The validated extend string.

    Raises
    ------
    ValueError
        If *extend* is not one of the four recognised modes.
    """
    if extend not in _VALID_EXTEND:
        raise ValueError(
            f"extend must be one of {_VALID_EXTEND!r}, got {extend!r}"
        )
    return extend


def _ensure_unit(value: Any, default_units: str) -> Unit:
    """Coerce *value* to a :class:`Unit` if it is not already one.

    Parameters
    ----------
    value : Any
        A :class:`Unit` instance, or a numeric scalar that will be
        wrapped in ``Unit(value, default_units)``.
    default_units : str
        Unit type used when *value* is not already a :class:`Unit`.

    Returns
    -------
    Unit
        The (possibly newly created) unit.
    """
    if is_unit(value):
        return value
    return Unit(value, default_units)


def _make_stops(
    colours: Sequence[str],
    stops: Optional[Sequence[float]],
) -> tuple[list[str], list[float]]:
    """Normalise *colours* and *stops* following R semantics.

    Both sequences are recycled to the length of the longer one.  If
    *stops* is ``None``, evenly-spaced values in [0, 1] are generated.

    Parameters
    ----------
    colours : Sequence[str]
        Colour specification strings.
    stops : Sequence[float] or None
        Gradient stop positions in [0, 1].

    Returns
    -------
    tuple[list[str], list[float]]
        ``(colours, stops)`` recycled to the same length.

    Raises
    ------
    ValueError
        If the resulting length is less than 1.
    """
    n_colours = len(colours)
    if stops is None:
        n_stops = n_colours
        stops_arr: np.ndarray = np.linspace(0.0, 1.0, n_stops)
    else:
        stops_arr = np.asarray(stops, dtype=float)
        n_stops = len(stops_arr)

    nstops = max(n_colours, n_stops)
    if nstops < 1:
        raise ValueError("colours and stops must be at least length 1")

    # Recycle both to *nstops* (mirroring R's ``rep(x, length.out=n)``).
    colours_out: list[str] = [
        colours[i % n_colours] for i in range(nstops)
    ]
    if len(stops_arr) < nstops:
        stops_arr = np.resize(stops_arr, nstops)
    stops_out: list[float] = stops_arr[:nstops].tolist()

    return colours_out, stops_out


# ======================================================================
# LinearGradient
# ======================================================================


class LinearGradient:
    """A linear colour gradient defined by two endpoints.

    This corresponds to R's ``grid::linearGradient()`` and the internal
    class ``GridLinearGradient``.

    Parameters
    ----------
    colours : list[str]
        Colour strings (e.g. ``["black", "white"]``).
    stops : list[float] or None
        Gradient stop positions in [0, 1].  ``None`` (default) produces
        evenly-spaced stops matching the length of *colours*.
    x1 : Unit or float or None
        Horizontal start of the gradient line.  Defaults to
        ``Unit(0, "npc")``.
    y1 : Unit or float or None
        Vertical start of the gradient line.  Defaults to
        ``Unit(0, "npc")``.
    x2 : Unit or float or None
        Horizontal end of the gradient line.  Defaults to
        ``Unit(1, "npc")``.
    y2 : Unit or float or None
        Vertical end of the gradient line.  Defaults to
        ``Unit(1, "npc")``.
    default_units : str
        Unit type applied when a coordinate is given as a plain number.
    extend : str
        One of ``"pad"``, ``"repeat"``, ``"reflect"``, or ``"none"``.
    group : bool
        If ``True`` the gradient is resolved relative to the bounding
        box of *all* shapes; if ``False`` it is resolved per shape.

    Raises
    ------
    ValueError
        If *extend* is invalid, or *colours*/*stops* have length < 1,
        or any coordinate has length != 1.
    """

    def __init__(
        self,
        colours: list[str],
        stops: Optional[list[float]] = None,
        x1: Optional[Union[Unit, float]] = None,
        y1: Optional[Union[Unit, float]] = None,
        x2: Optional[Union[Unit, float]] = None,
        y2: Optional[Union[Unit, float]] = None,
        default_units: str = "npc",
        extend: str = "pad",
        group: bool = True,
    ) -> None:
        self.colours, self.stops = _make_stops(colours, stops)

        self.x1: Unit = _ensure_unit(
            x1 if x1 is not None else 0.0, default_units
        )
        self.y1: Unit = _ensure_unit(
            y1 if y1 is not None else 0.0, default_units
        )
        self.x2: Unit = _ensure_unit(
            x2 if x2 is not None else 1.0, default_units
        )
        self.y2: Unit = _ensure_unit(
            y2 if y2 is not None else 1.0, default_units
        )

        # Each coordinate must be scalar (length 1).
        for name, val in (
            ("x1", self.x1),
            ("y1", self.y1),
            ("x2", self.x2),
            ("y2", self.y2),
        ):
            if len(val) != 1:
                raise ValueError(
                    f"{name} must be length 1, got length {len(val)}"
                )

        self.extend: str = _validate_extend(extend)
        self.group: bool = bool(group)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"LinearGradient(colours={self.colours!r}, "
            f"stops={self.stops!r}, "
            f"x1={self.x1!r}, y1={self.y1!r}, "
            f"x2={self.x2!r}, y2={self.y2!r}, "
            f"extend={self.extend!r}, group={self.group!r})"
        )


# ======================================================================
# RadialGradient
# ======================================================================


class RadialGradient:
    """A radial colour gradient defined by two circles.

    This corresponds to R's ``grid::radialGradient()`` and the internal
    class ``GridRadialGradient``.

    Parameters
    ----------
    colours : list[str]
        Colour strings.
    stops : list[float] or None
        Gradient stop positions in [0, 1].
    cx1 : Unit or float or None
        Horizontal centre of the inner circle.  Default ``0.5 npc``.
    cy1 : Unit or float or None
        Vertical centre of the inner circle.  Default ``0.5 npc``.
    r1 : Unit or float or None
        Radius of the inner circle.  Default ``0 npc``.
    cx2 : Unit or float or None
        Horizontal centre of the outer circle.  Default ``0.5 npc``.
    cy2 : Unit or float or None
        Vertical centre of the outer circle.  Default ``0.5 npc``.
    r2 : Unit or float or None
        Radius of the outer circle.  Default ``0.5 npc``.
    default_units : str
        Unit type applied when a parameter is given as a plain number.
    extend : str
        One of ``"pad"``, ``"repeat"``, ``"reflect"``, or ``"none"``.
    group : bool
        If ``True`` the gradient is resolved relative to the bounding
        box of *all* shapes; if ``False`` it is resolved per shape.

    Raises
    ------
    ValueError
        If *extend* is invalid, colours/stops have length < 1, or any
        coordinate/radius has length != 1.
    """

    def __init__(
        self,
        colours: list[str],
        stops: Optional[list[float]] = None,
        cx1: Optional[Union[Unit, float]] = None,
        cy1: Optional[Union[Unit, float]] = None,
        r1: Optional[Union[Unit, float]] = None,
        cx2: Optional[Union[Unit, float]] = None,
        cy2: Optional[Union[Unit, float]] = None,
        r2: Optional[Union[Unit, float]] = None,
        default_units: str = "npc",
        extend: str = "pad",
        group: bool = True,
    ) -> None:
        self.colours, self.stops = _make_stops(colours, stops)

        self.cx1: Unit = _ensure_unit(
            cx1 if cx1 is not None else 0.5, default_units
        )
        self.cy1: Unit = _ensure_unit(
            cy1 if cy1 is not None else 0.5, default_units
        )
        self.r1: Unit = _ensure_unit(
            r1 if r1 is not None else 0.0, default_units
        )
        self.cx2: Unit = _ensure_unit(
            cx2 if cx2 is not None else 0.5, default_units
        )
        self.cy2: Unit = _ensure_unit(
            cy2 if cy2 is not None else 0.5, default_units
        )
        self.r2: Unit = _ensure_unit(
            r2 if r2 is not None else 0.5, default_units
        )

        for name, val in (
            ("cx1", self.cx1),
            ("cy1", self.cy1),
            ("r1", self.r1),
            ("cx2", self.cx2),
            ("cy2", self.cy2),
            ("r2", self.r2),
        ):
            if len(val) != 1:
                raise ValueError(
                    f"{name} must be length 1, got length {len(val)}"
                )

        self.extend: str = _validate_extend(extend)
        self.group: bool = bool(group)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RadialGradient(colours={self.colours!r}, "
            f"stops={self.stops!r}, "
            f"cx1={self.cx1!r}, cy1={self.cy1!r}, r1={self.r1!r}, "
            f"cx2={self.cx2!r}, cy2={self.cy2!r}, r2={self.r2!r}, "
            f"extend={self.extend!r}, group={self.group!r})"
        )


# ======================================================================
# Pattern (tiling pattern)
# ======================================================================

# Justification helpers -- resolve a single *just* string to (hjust, vjust).
_JUST_H: dict[str, float] = {
    "left": 0.0,
    "right": 1.0,
    "centre": 0.5,
    "center": 0.5,
    "top": 0.5,
    "bottom": 0.5,
}

_JUST_V: dict[str, float] = {
    "left": 0.5,
    "right": 0.5,
    "centre": 0.5,
    "center": 0.5,
    "top": 1.0,
    "bottom": 0.0,
}


def _resolve_just(
    just: Union[str, tuple[float, float]],
) -> tuple[float, float]:
    """Return ``(hjust, vjust)`` from a justification specification.

    Parameters
    ----------
    just : str or tuple[float, float]
        A justification string (``"centre"``, ``"left"``, etc.) or an
        explicit ``(hjust, vjust)`` pair.

    Returns
    -------
    tuple[float, float]
        Numeric ``(hjust, vjust)`` in [0, 1].

    Raises
    ------
    ValueError
        If *just* is an unrecognised string.
    """
    if isinstance(just, str):
        j = just.lower()
        if j not in _JUST_H:
            raise ValueError(
                f"Unrecognised justification string: {just!r}"
            )
        return _JUST_H[j], _JUST_V[j]
    # Assume numeric pair.
    return (float(just[0]), float(just[1]))


class Pattern:
    """A tiling pattern fill based on an arbitrary grob.

    This corresponds to R's ``grid::pattern()`` and the internal class
    ``GridTilingPattern``.

    Parameters
    ----------
    grob : Any
        A grob (graphical object) to use as the repeating tile.
    x : Unit or float or None
        Horizontal position of the tile.  Default ``0.5 npc``.
    y : Unit or float or None
        Vertical position of the tile.  Default ``0.5 npc``.
    width : Unit or float or None
        Width of the tile.  Default ``1 npc``.
    height : Unit or float or None
        Height of the tile.  Default ``1 npc``.
    default_units : str
        Unit type applied when a dimension is given as a plain number.
    just : str or tuple[float, float]
        Justification of the tile relative to ``(x, y)``.  Accepts
        standard strings such as ``"centre"``, ``"left"``, etc., or an
        explicit ``(hjust, vjust)`` pair.
    extend : str
        One of ``"pad"``, ``"repeat"``, ``"reflect"``, or ``"none"``.
    group : bool
        If ``True`` the pattern is resolved relative to the bounding
        box of *all* shapes; if ``False`` it is resolved per shape.

    Raises
    ------
    ValueError
        If *extend* is invalid or any coordinate/dimension has
        length != 1.
    """

    def __init__(
        self,
        grob: Any,
        x: Optional[Union[Unit, float]] = None,
        y: Optional[Union[Unit, float]] = None,
        width: Optional[Union[Unit, float]] = None,
        height: Optional[Union[Unit, float]] = None,
        default_units: str = "npc",
        just: Union[str, tuple[float, float]] = "centre",
        extend: str = "pad",
        group: bool = True,
    ) -> None:
        self.grob: Any = grob

        self.x: Unit = _ensure_unit(
            x if x is not None else 0.5, default_units
        )
        self.y: Unit = _ensure_unit(
            y if y is not None else 0.5, default_units
        )
        self.width: Unit = _ensure_unit(
            width if width is not None else 1.0, default_units
        )
        self.height: Unit = _ensure_unit(
            height if height is not None else 1.0, default_units
        )

        for name, val in (
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ):
            if len(val) != 1:
                raise ValueError(
                    f"{name} must be length 1, got length {len(val)}"
                )

        self.hjust: float
        self.vjust: float
        self.hjust, self.vjust = _resolve_just(just)

        self.extend: str = _validate_extend(extend)
        self.group: bool = bool(group)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"Pattern(grob={self.grob!r}, "
            f"x={self.x!r}, y={self.y!r}, "
            f"width={self.width!r}, height={self.height!r}, "
            f"hjust={self.hjust!r}, vjust={self.vjust!r}, "
            f"extend={self.extend!r}, group={self.group!r})"
        )


# ======================================================================
# Factory functions
# ======================================================================


def linear_gradient(
    colours: list[str] = None,
    stops: Optional[list[float]] = None,
    x1: Optional[Union[Unit, float]] = None,
    y1: Optional[Union[Unit, float]] = None,
    x2: Optional[Union[Unit, float]] = None,
    y2: Optional[Union[Unit, float]] = None,
    default_units: str = "npc",
    extend: str = "pad",
    group: bool = True,
) -> LinearGradient:
    """Create a :class:`LinearGradient`.

    This is a convenience wrapper matching R's
    ``grid::linearGradient()`` function signature.

    Parameters
    ----------
    colours : list[str], optional
        Colour strings.  Defaults to ``["black", "white"]``.
    stops : list[float] or None
        Gradient stop positions in [0, 1].
    x1 : Unit or float or None
        Horizontal start of the gradient line.
    y1 : Unit or float or None
        Vertical start of the gradient line.
    x2 : Unit or float or None
        Horizontal end of the gradient line.
    y2 : Unit or float or None
        Vertical end of the gradient line.
    default_units : str
        Unit type used for bare numeric coordinates.
    extend : str
        Gradient extension mode.
    group : bool
        Resolve gradient relative to all shapes (``True``) or per
        shape (``False``).

    Returns
    -------
    LinearGradient
        A new linear gradient object.

    Examples
    --------
    >>> lg = linear_gradient(["red", "blue"])
    >>> lg.colours
    ['red', 'blue']
    """
    if colours is None:
        colours = ["black", "white"]
    return LinearGradient(
        colours=colours,
        stops=stops,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        default_units=default_units,
        extend=extend,
        group=group,
    )


def radial_gradient(
    colours: list[str] = None,
    stops: Optional[list[float]] = None,
    cx1: Optional[Union[Unit, float]] = None,
    cy1: Optional[Union[Unit, float]] = None,
    r1: Optional[Union[Unit, float]] = None,
    cx2: Optional[Union[Unit, float]] = None,
    cy2: Optional[Union[Unit, float]] = None,
    r2: Optional[Union[Unit, float]] = None,
    default_units: str = "npc",
    extend: str = "pad",
    group: bool = True,
) -> RadialGradient:
    """Create a :class:`RadialGradient`.

    This is a convenience wrapper matching R's
    ``grid::radialGradient()`` function signature.

    Parameters
    ----------
    colours : list[str], optional
        Colour strings.  Defaults to ``["black", "white"]``.
    stops : list[float] or None
        Gradient stop positions in [0, 1].
    cx1 : Unit or float or None
        Horizontal centre of the inner circle.
    cy1 : Unit or float or None
        Vertical centre of the inner circle.
    r1 : Unit or float or None
        Radius of the inner circle.
    cx2 : Unit or float or None
        Horizontal centre of the outer circle.
    cy2 : Unit or float or None
        Vertical centre of the outer circle.
    r2 : Unit or float or None
        Radius of the outer circle.
    default_units : str
        Unit type used for bare numeric values.
    extend : str
        Gradient extension mode.
    group : bool
        Resolve gradient relative to all shapes (``True``) or per
        shape (``False``).

    Returns
    -------
    RadialGradient
        A new radial gradient object.

    Examples
    --------
    >>> rg = radial_gradient(["white", "black"])
    >>> rg.r2
    Unit([0.5], ['npc'])
    """
    if colours is None:
        colours = ["black", "white"]
    return RadialGradient(
        colours=colours,
        stops=stops,
        cx1=cx1,
        cy1=cy1,
        r1=r1,
        cx2=cx2,
        cy2=cy2,
        r2=r2,
        default_units=default_units,
        extend=extend,
        group=group,
    )


def pattern(
    grob: Any,
    x: Optional[Union[Unit, float]] = None,
    y: Optional[Union[Unit, float]] = None,
    width: Optional[Union[Unit, float]] = None,
    height: Optional[Union[Unit, float]] = None,
    default_units: str = "npc",
    just: Union[str, tuple[float, float]] = "centre",
    extend: str = "pad",
    group: bool = True,
) -> Pattern:
    """Create a :class:`Pattern` (tiling fill).

    This is a convenience wrapper matching R's ``grid::pattern()``
    function signature.

    Parameters
    ----------
    grob : Any
        A grob to use as the repeating tile.
    x : Unit or float or None
        Horizontal position of the tile.
    y : Unit or float or None
        Vertical position of the tile.
    width : Unit or float or None
        Width of the tile.
    height : Unit or float or None
        Height of the tile.
    default_units : str
        Unit type used for bare numeric values.
    just : str or tuple[float, float]
        Tile justification relative to ``(x, y)``.
    extend : str
        Pattern extension mode.
    group : bool
        Resolve pattern relative to all shapes (``True``) or per
        shape (``False``).

    Returns
    -------
    Pattern
        A new tiling pattern object.

    Examples
    --------
    >>> pat = pattern("placeholder_grob")
    >>> pat.hjust
    0.5
    """
    return Pattern(
        grob=grob,
        x=x,
        y=y,
        width=width,
        height=height,
        default_units=default_units,
        just=just,
        extend=extend,
        group=group,
    )


# ============================================================================
# Pattern resolution pipeline
# Port of R patterns.R:140-429
# ============================================================================


def is_pattern(fill: Any) -> bool:
    """Test whether *fill* is a grid pattern (gradient or tiling).

    Mirrors R ``is.pattern()``.
    """
    return isinstance(fill, (LinearGradient, RadialGradient, Pattern))


def is_pattern_list(fill: Any) -> bool:
    """Test whether *fill* is a list of patterns."""
    if isinstance(fill, (list, tuple)):
        return len(fill) > 0 and all(is_pattern(f) for f in fill)
    return False


def is_resolved_pattern(fill: Any) -> bool:
    """Test whether a pattern has already been resolved."""
    return isinstance(fill, dict) and fill.get("_resolved", False)


class ResolvedPattern:
    """A pattern that has been resolved to renderer-specific form.

    Port of R ``resolvedPattern()`` (patterns.R:192-196).
    Wraps the original pattern with a ``ref`` (renderer handle) and
    marks it as resolved.
    """

    def __init__(self, pattern: Any, ref: Any) -> None:
        self.pattern = pattern
        self.ref = ref
        self._resolved = True

    def __repr__(self) -> str:
        return f"ResolvedPattern(ref={self.ref!r})"


# ---------------------------------------------------------------------------
# resolveFill -- S3-like dispatcher
# Port of R patterns.R:198-385
# ---------------------------------------------------------------------------


def resolve_fill(fill: Any, index: int = 1, grob: Any = None) -> Any:
    """Resolve a fill value to a renderer-ready form.

    Port of R ``resolveFill()`` (patterns.R:198-385).
    Dispatches based on fill type:
    - Simple colour strings pass through unchanged
    - Already-resolved patterns pass through
    - Unresolved patterns → resolvePattern()
    - Pattern lists → resolve each element

    Parameters
    ----------
    fill : Any
        Fill value from gpar. May be a string, LinearGradient,
        RadialGradient, Pattern, ResolvedPattern, list, etc.
    index : int
        Shape index (1-based, for per-shape pattern lists).
    grob : Grob or None
        The grob being filled (attached for coordinate queries).

    Returns
    -------
    Any
        Resolved fill value (string, ResolvedPattern, or "transparent").
    """
    # Already resolved (R patterns.R:210-212)
    if isinstance(fill, ResolvedPattern):
        return fill

    # Simple fill (R patterns.R:205-207)
    if not is_pattern(fill) and not is_pattern_list(fill):
        return fill

    # Single pattern (R patterns.R:217-280)
    if is_pattern(fill):
        return _resolve_fill_pattern(fill, index, grob)

    # Pattern list (R patterns.R:222-334)
    if is_pattern_list(fill):
        return _resolve_fill_pattern_list(fill, grob)

    return fill


def _resolve_fill_pattern(fill: Any, index: int, grob: Any) -> Any:
    """Resolve a single pattern fill.

    Port of R ``resolveFill.GridGrobPattern`` (patterns.R:237-280).
    """
    from ._coords import grob_points, is_empty_coords, coords_bbox
    from ._viewport import Viewport, push_viewport, pop_viewport
    from ._gpar import Gpar

    # If grob is available, compute bounding box
    if grob is not None:
        pts = grob_points(grob, closed=True)
        if not is_empty_coords(pts):
            if getattr(fill, "group", True) or len(pts) <= 1:
                bbox = coords_bbox(pts)
            else:
                # Per-shape bounding box
                idx = (index - 1) % len(pts)
                bbox = coords_bbox(pts[idx] if hasattr(pts, '__getitem__') else pts)

            # Push temporary viewport for pattern resolution
            # (R patterns.R:266-271)
            vp = Viewport(
                x=bbox["left"], y=bbox["bottom"],
                width=bbox["width"], height=bbox["height"],
                default_units="inches",
                just=("left", "bottom"),
                clip="off", mask="none",
            )
            push_viewport(vp, recording=False)
            result = resolve_pattern(fill)
            pop_viewport(recording=False)
            return result
        else:
            import warnings
            warnings.warn("Pattern fill applied to object with no inside")
            return "transparent"

    # No grob context — resolve in current viewport
    return resolve_pattern(fill)


def _resolve_fill_pattern_list(fill: list, grob: Any) -> Any:
    """Resolve a list of patterns.

    Port of R ``resolveFill.GridGrobPatternList`` (patterns.R:283-334).
    """
    resolved = []
    for i, f in enumerate(fill):
        if isinstance(f, ResolvedPattern):
            resolved.append(f)
        elif is_pattern(f):
            resolved.append(_resolve_fill_pattern(f, i + 1, grob))
        else:
            resolved.append(f)
    return resolved


# ---------------------------------------------------------------------------
# resolvePattern -- type-specific resolution
# Port of R patterns.R:387-429
# ---------------------------------------------------------------------------


def resolve_pattern(pattern: Any) -> Any:
    """Resolve a pattern object to renderer-specific form.

    Port of R ``resolvePattern()`` (patterns.R:387-429).
    Dispatches based on pattern type: LinearGradient, RadialGradient,
    or Pattern (tiling).

    The resolved pattern contains a renderer-specific ``ref`` that can
    be used directly by the rendering backend.

    Parameters
    ----------
    pattern : LinearGradient or RadialGradient or Pattern
        The pattern to resolve.

    Returns
    -------
    ResolvedPattern
        The pattern with renderer-specific reference attached.
    """
    if isinstance(pattern, LinearGradient):
        return _resolve_linear_gradient(pattern)
    elif isinstance(pattern, RadialGradient):
        return _resolve_radial_gradient(pattern)
    elif isinstance(pattern, Pattern):
        return _resolve_tiling_pattern(pattern)
    return pattern


def _resolve_linear_gradient(grad: LinearGradient) -> ResolvedPattern:
    """Resolve a linear gradient to renderer-specific form.

    Port of R ``resolvePattern.GridLinearGradient`` (patterns.R:391-399).
    Converts endpoints to device coordinates, creates a renderer pattern.
    """
    # The renderer will handle the actual gradient creation when it
    # encounters this object in gpar$fill.  We store the resolved
    # coordinates for the renderer to use.
    from ._units import device_loc

    try:
        p1 = device_loc(grad.x1, grad.y1, value_only=True, device=True)
        p2 = device_loc(grad.x2, grad.y2, value_only=True, device=True)
        ref = {
            "type": "linear_gradient",
            "x1": float(p1["x"][0]), "y1": float(p1["y"][0]),
            "x2": float(p2["x"][0]), "y2": float(p2["y"][0]),
            "colours": grad.colours,
            "stops": grad.stops,
            "extend": grad.extend,
        }
    except (ValueError, AttributeError, TypeError, IndexError):
        # Documented deferred-resolution fallback: when the gradient's
        # endpoint Units can't be resolved against the current
        # renderer state (e.g. ``"npc"`` outside a viewport), stash the
        # original gradient so the renderer can resolve it later. R's
        # ``resolvePattern.GridLinearGradient`` (patterns.R:401-418)
        # delegates the same way via lazy attr lookup.
        ref = {"type": "linear_gradient", "pattern": grad}

    return ResolvedPattern(grad, ref)


def _resolve_radial_gradient(grad: RadialGradient) -> ResolvedPattern:
    """Resolve a radial gradient to renderer-specific form.

    Port of R ``resolvePattern.GridRadialGradient`` (patterns.R:401-418).
    """
    from ._units import device_loc, device_dim, Unit
    import math

    try:
        c1 = device_loc(grad.cx1, grad.cy1, value_only=True, device=True)
        c2 = device_loc(grad.cx2, grad.cy2, value_only=True, device=True)
        # R computes r as min of two axis scalings (patterns.R:403-411)
        dim1a = device_dim(Unit(0, "inches"), grad.r1, value_only=True, device=True)
        dim1b = device_dim(grad.r1, Unit(0, "inches"), value_only=True, device=True)
        r1 = min(math.sqrt(dim1a["w"][0]**2 + dim1a["h"][0]**2),
                 math.sqrt(dim1b["w"][0]**2 + dim1b["h"][0]**2))
        dim2a = device_dim(Unit(0, "inches"), grad.r2, value_only=True, device=True)
        dim2b = device_dim(grad.r2, Unit(0, "inches"), value_only=True, device=True)
        r2 = min(math.sqrt(dim2a["w"][0]**2 + dim2a["h"][0]**2),
                 math.sqrt(dim2b["w"][0]**2 + dim2b["h"][0]**2))
        ref = {
            "type": "radial_gradient",
            "cx1": float(c1["x"][0]), "cy1": float(c1["y"][0]), "r1": r1,
            "cx2": float(c2["x"][0]), "cy2": float(c2["y"][0]), "r2": r2,
            "colours": grad.colours,
            "stops": grad.stops,
            "extend": grad.extend,
        }
    except (ValueError, AttributeError, TypeError, IndexError):
        # Deferred-resolution fallback — see ``_resolve_linear_gradient``.
        ref = {"type": "radial_gradient", "pattern": grad}

    return ResolvedPattern(grad, ref)


def _resolve_tiling_pattern(pat: Pattern) -> ResolvedPattern:
    """Resolve a tiling pattern to renderer-specific form.

    Port of R ``resolvePattern.GridTilingPattern`` (patterns.R:420-429).
    """
    from ._units import device_loc, device_dim

    try:
        xy = device_loc(pat.x, pat.y, value_only=True, device=True)
        wh = device_dim(pat.width, pat.height, value_only=True, device=True)
        left = float(xy["x"][0]) - pat.hjust * float(wh["w"][0])
        bottom = float(xy["y"][0]) - pat.vjust * float(wh["h"][0])
        ref = {
            "type": "tiling_pattern",
            "grob": pat.grob,
            "x": left, "y": bottom,
            "width": float(wh["w"][0]), "height": float(wh["h"][0]),
            "extend": pat.extend,
        }
    except (ValueError, AttributeError, TypeError, IndexError):
        # Deferred-resolution fallback — see ``_resolve_linear_gradient``.
        ref = {"type": "tiling_pattern", "pattern": pat}

    return ResolvedPattern(pat, ref)


# ---------------------------------------------------------------------------
# recordGrobForPatternResolution / recordGTreeForPatternResolution
# Port of R patterns.R:150-190
# ---------------------------------------------------------------------------


def record_grob_for_pattern_resolution(grob: Any) -> None:
    """Attach the built grob to gpar$fill for pattern resolution.

    Port of R ``recordGrobForPatternResolution`` (patterns.R:150-161).
    Called after ``makeContent()`` in the drawing pipeline.
    If the current gpar has a pattern fill, the grob is attached so
    that subsequent ``resolveFill()`` can compute the bounding box.
    """
    from ._state import get_state

    state = get_state()
    gp = state.get_gpar()
    if gp is None:
        return

    fill = gp.get("fill", None)
    if fill is None:
        return

    if is_pattern(fill):
        # Attach grob to the pattern for later resolution
        fill._attached_grob = grob
    elif is_pattern_list(fill):
        for f in fill:
            if is_pattern(f):
                f._attached_grob = grob


def record_gtree_for_pattern_resolution(grob: Any) -> None:
    """Resolve gTree-level pattern fill immediately.

    Port of R ``recordGTreeForPatternResolution`` (patterns.R:176-190).
    If the gTree's own gp$fill is a pattern with group=TRUE,
    resolve it now (using the gTree's bounding box).
    """
    from ._state import get_state

    gp = getattr(grob, "gp", None)
    if gp is None:
        return

    fill = gp.get("fill", None) if hasattr(gp, "get") else None
    if fill is None:
        return

    should_resolve = False
    if is_pattern(fill) and getattr(fill, "group", False):
        should_resolve = True
    elif is_pattern_list(fill):
        should_resolve = True

    if should_resolve:
        state = get_state()
        current_gp = state.get_gpar()
        current_fill = current_gp.get("fill", None) if current_gp else None
        if current_fill is not None and (is_pattern(current_fill) or is_pattern_list(current_fill)):
            resolved = resolve_fill(current_fill, grob=grob)
            if resolved is None:
                current_gp._params["fill"] = "transparent"
            else:
                current_gp._params["fill"] = resolved
