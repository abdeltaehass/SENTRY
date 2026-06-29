"""Multi-image (multi-view) input support.

A scene may be covered by several cameras / angles. BLIP-2 takes a single image,
so we compose the available views into one panel image the existing pipeline can
consume — no model surgery. (Zero-shot unless the model is trained on composites.)

A record keeps every view of a scene in `image_paths`.
"""

from __future__ import annotations

from PIL import Image


def compose_views(image_paths, layout: str = "horizontal", size: int = 224,
                  bg=(0, 0, 0)) -> Image.Image:
    """Compose 1+ view images into a single RGB panel (side-by-side or stacked)."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    if len(images) == 1:
        return images[0]

    tiles = [im.resize((size, size)) for im in images]
    n = len(tiles)
    if layout == "horizontal":
        canvas = Image.new("RGB", (size * n, size), bg)
        for i, tile in enumerate(tiles):
            canvas.paste(tile, (i * size, 0))
    else:  # vertical
        canvas = Image.new("RGB", (size, size * n), bg)
        for i, tile in enumerate(tiles):
            canvas.paste(tile, (0, i * size))
    return canvas


def record_image(record: dict, multiview: bool = False, **compose_kwargs) -> Image.Image:
    """Image for a record: primary view, or a composite of all views."""
    if multiview and record.get("image_paths"):
        return compose_views(record["image_paths"], **compose_kwargs)
    return Image.open(record["image_path"]).convert("RGB")
