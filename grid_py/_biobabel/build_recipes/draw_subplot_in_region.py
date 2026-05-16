"""
build_recipe_id: grid_py.draw_subplot_in_region
idiom: push_draw_pop
concepts: [Viewport, Unit, Grob]

The minimal build-recipe for the push-draw-pop idiom. A build-recipe is
intentionally smaller than a task recipe — it demonstrates exactly one idiom
so an agent can pattern-match it.
"""

from grid_py import (
    Unit,
    Viewport,
    gpar,
    grid_rect,
    pop_viewport,
    push_viewport,
)

push_viewport(Viewport(
    x=Unit(0.5, "npc"),
    y=Unit(0.5, "npc"),
    width=Unit(0.4, "npc"),
    height=Unit(0.4, "npc"),
))
try:
    grid_rect(gp=gpar(fill="lightcoral"))
finally:
    pop_viewport()
