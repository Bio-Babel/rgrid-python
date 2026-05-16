"""
build_recipe_id: grid_py.make_legend_block
idiom: build_grobtree
concepts: [Grob, Viewport, Unit]

Build a 4-row legend (color swatch + label) as a single grob_tree, then draw
it inside a small viewport at the top-right corner of the device.
"""

from grid_py import (
    Unit,
    Viewport,
    gpar,
    grid_draw,
    grob_tree,
    pop_viewport,
    push_viewport,
    rect_grob,
    text_grob,
)


labels_and_colors = [
    ("up-regulated", "tomato"),
    ("down-regulated", "steelblue"),
    ("ambiguous", "grey60"),
    ("filtered", "grey85"),
]

n = len(labels_and_colors)
row_height = 1.0 / n

children: list = []
for i, (label, fill) in enumerate(labels_and_colors):
    y_center = (n - i - 0.5) * row_height
    children.append(rect_grob(
        x=Unit(0.15, "npc"),
        y=Unit(y_center, "npc"),
        width=Unit(0.18, "npc"),
        height=Unit(0.6 * row_height, "npc"),
        gp=gpar(fill=fill, col=None),
    ))
    children.append(text_grob(
        label,
        x=Unit(0.36, "npc"),
        y=Unit(y_center, "npc"),
        hjust=0,
    ))

legend = grob_tree(*children)

push_viewport(Viewport(
    x=Unit(0.82, "npc"),
    y=Unit(0.82, "npc"),
    width=Unit(0.28, "npc"),
    height=Unit(0.28, "npc"),
))
try:
    grid_draw(legend)
finally:
    pop_viewport()
