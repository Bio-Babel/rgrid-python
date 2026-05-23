"""Linetype resolution — R-faithful port of GE_LTYpar + CairoLineType.

This is the single source of truth for translating any R-accepted lty
specification (named string, R integer 0..6, hex string of length
{2,4,6,8}) into a Cairo / SVG ready dash array.  Both the Cairo and the
Web renderers consume the same ``resolve_lty`` output, so any lty
behaviour fix lands in exactly one place.

R reference (R 4.5.3):
    src/include/R_ext/GraphicsEngine.h:406-412   (LTY_* constants)
    src/main/engine.c::GE_LTYpar                 (string -> 32-bit int)
    src/library/grDevices/src/cairoFns.c::CairoLineType
                                                 (int -> dash array)

The cairo-dash scaling factor is *1.0*: ``set_dash([nibble * lwd, ...])``.
This was empirically verified by reverse-engineering R's cairo PNG output
(lty="44" at lwd in {1, 1.5, 2, 4, 6, 8} matches ``period = 2*nibble*lwd
+ 2*cap``, where cap ~= lwd for the default lineend="round").  A 0.75
factor that appeared in early drafts of this file was a memory error
(possibly conflated with a non-cairo R device) and would cause every
dash pattern to shrink by 25 %.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np

__all__ = [
    "resolve_lty",
    "is_blank_lty",
    "valid_named_lty",
]


# ---------------------------------------------------------------------------
# R-gold constants (verbatim from R source, do not edit without R reference)
# ---------------------------------------------------------------------------

# GraphicsEngine.h:407-412 — every named lty is *also* expressible as a
# hex string, so resolve_lty has a single parser code path.  Low nibble
# of the integer constant sits in the leftmost character of the hex
# string (R's GE_LTYpar packs bits low-to-high as it scans left-to-right).
#   LTY_SOLID    = 0          ""
#   LTY_DASHED   = 0x44       "44"
#   LTY_DOTTED   = 0x31       -> low nibble 1, high nibble 3 -> "13"
#   LTY_DOTDASH  = 0x3431                                     "1343"
#   LTY_LONGDASH = 0x37                                       "73"
#   LTY_TWODASH  = 0x2622                                     "2262"
_LTY_NAMED_TO_HEX: dict[str, str] = {
    "solid":    "",
    "dashed":   "44",
    "dotted":   "13",
    "dotdash":  "1343",
    "longdash": "73",
    "twodash":  "2262",
}

# R ?par integer codes (user-facing).  Note that "user-facing 0" is BLANK,
# *not* the internal LTY_SOLID=0 constant: GE_LTYpar maps user 0 -> internal
# LTY_BLANK (-1) and user 1 -> internal LTY_SOLID (0).
_LTY_INT_TO_NAME: dict[int, str] = {
    1: "solid",
    2: "dashed",
    3: "dotted",
    4: "dotdash",
    5: "longdash",
    6: "twodash",
}

# R-gold (empirically verified): GE_LTYpar rejects any hex string whose
# length is not in {2, 4, 6, 8}.  Single nibble "F" -> "invalid line
# type: must be length 2, 4, 6 or 8".
_LTY_VALID_HEX_LENS: frozenset[int] = frozenset({2, 4, 6, 8})

LtyInput = Union[str, int, float, None, list, tuple, np.ndarray]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def valid_named_lty() -> frozenset[str]:
    """Set of R named lty values accepted by Gpar (includes ``"blank"``)."""
    return frozenset(_LTY_NAMED_TO_HEX.keys()) | {"blank"}


def is_blank_lty(value: LtyInput) -> bool:
    """True iff *value* represents R LTY_BLANK (caller should skip stroke).

    Mirrors R par convention:
      * user-facing integer 0   -> blank
      * "blank" (string)        -> blank
      * NA / None inside a Gpar list sentinel  -> blank
      * **but** scalar ``None`` (no lty supplied) -> not blank (= solid)
      * **but** string ``"0"`` -> not blank (R rejects len-1 hex; see
        resolve_lty); we follow R's quirk and only treat *integer* 0 as
        blank, not string "0".
    """
    if value is None:
        return False
    raw = value[0] if isinstance(value, (list, tuple, np.ndarray)) and len(value) else value
    if raw is None:
        return True
    if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, bool):
        return int(raw) == 0
    return str(raw).lower() == "blank"


def resolve_lty(value: LtyInput, lwd: float = 1.0) -> Optional[list[float]]:
    """Resolve any R-accepted lty input into a Cairo-ready dash array.

    Parameters
    ----------
    value : str | int | float | None | sequence
        - ``None``                       -> solid
        - One of ``valid_named_lty()``   -> the canonical hex pattern
        - Hex string, length in {2,4,6,8} with chars [0-9A-Fa-f]
        - Integer in 1..6 (R par convention; 0 is BLANK and must be
          caught by ``is_blank_lty`` *before* calling this function)
        - List/tuple/ndarray: first element is used (R recycling rule
          is the caller's job)

    Returns
    -------
    None or list[float]
        - ``None`` for solid (no dashing)
        - Otherwise the cairo ``set_dash`` array.  Each element is
          ``hex_digit * lwd`` user-space units (or whatever space the
          caller's ``lwd`` lives in — caller must keep lwd and dash in
          the same coordinate system).

    Raises
    ------
    ValueError
        On unrecognised input.  Caller must invoke ``is_blank_lty``
        first; passing a blank-representing value here raises.
    """
    if value is None:
        return None
    raw = value[0] if isinstance(value, (list, tuple, np.ndarray)) and len(value) else value
    if raw is None:
        # Caller forgot to short-circuit via is_blank_lty.  Fail loud
        # (per principle 4 — do not silently convert blank to solid).
        raise ValueError(
            "blank lty (NA sentinel) must be handled by caller via is_blank_lty()"
        )

    # R integer path — par convention 1..6 (0 is BLANK, handled by caller).
    if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, bool):
        i = int(raw)
        if i == 0:
            raise ValueError(
                "lty=0 is BLANK; caller must check is_blank_lty() before resolve_lty()"
            )
        if i not in _LTY_INT_TO_NAME:
            raise ValueError(f"Invalid integer lty: {i!r} (must be in 1..6)")
        raw = _LTY_INT_TO_NAME[i]

    s = str(raw)

    # Named lty -> canonical hex (single parser path below for both forms)
    if s in _LTY_NAMED_TO_HEX:
        s = _LTY_NAMED_TO_HEX[s]

    if s == "":
        return None   # solid

    # Hex string validation — R GE_LTYpar rules (empirically verified):
    #   non-hex char        -> "invalid hex digit in 'color' or 'lty'"
    #   length not in 2/4/6/8 -> "invalid line type: must be length 2, 4, 6 or 8"
    if not all(c in "0123456789abcdefABCDEF" for c in s):
        raise ValueError(f"Invalid lty: {s!r} (invalid hex digit)")
    if len(s) not in _LTY_VALID_HEX_LENS:
        raise ValueError(
            f"Invalid lty: {s!r} (length {len(s)} not in {{2,4,6,8}})"
        )

    # Cairo dash expansion — equivalent to R cairoFns.c::CairoLineType:
    #     while (l < 8 && lty != 0) {
    #         dt = lty & 15;
    #         if (dt == 0) break;     // zero nibble terminates pattern
    #         ls[l] = dt * lwd;       // scale factor 1.0 (R-verified)
    #         lty = lty >> 4;
    #         l++;
    #     }
    # R's length pre-check guarantees s is len 2/4/6/8, so the only place a
    # zero nibble appears is at the *end* of a longer pattern (e.g. "4400"
    # legitimately means [4, 4] then stop).
    dashes: list[float] = []
    for c in s:
        d = int(c, 16)
        if d == 0:
            break
        dashes.append(d * lwd)
    return dashes if dashes else None
