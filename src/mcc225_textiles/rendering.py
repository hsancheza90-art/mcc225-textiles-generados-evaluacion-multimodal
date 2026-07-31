"""Renderizado determinista de patrones geométricos sintéticos."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image, ImageDraw, ImageFont


RGBColor = tuple[int, int, int]
Palette = tuple[RGBColor, ...]
DrawFunction = Callable[
    [
        ImageDraw.ImageDraw,
        Palette,
        int,
        int,
        random.Random,
    ],
    None,
]


def normalize_palette(
    raw_palette: Sequence[Sequence[int]],
) -> Palette:
    """Convierte y valida una paleta RGB."""

    palette: list[RGBColor] = []

    for color in raw_palette:
        if len(color) != 3:
            raise ValueError(
                "Cada color debe contener tres componentes RGB."
            )

        rgb = tuple(int(component) for component in color)

        if any(component < 0 or component > 255 for component in rgb):
            raise ValueError(
                f"Componente RGB fuera de rango: {rgb}"
            )

        palette.append(rgb)

    if len(palette) < 3:
        raise ValueError(
            "Se requieren al menos tres colores."
        )

    return tuple(palette)


def _diamond_points(
    center_x: int,
    center_y: int,
    radius: int,
) -> list[tuple[int, int]]:
    return [
        (center_x, center_y - radius),
        (center_x + radius, center_y),
        (center_x, center_y + radius),
        (center_x - radius, center_y),
    ]


def _draw_horizontal_bands(
    draw: ImageDraw.ImageDraw,
    palette: Palette,
    width: int,
    height: int,
    rng: random.Random,
) -> None:
    del rng

    band_height = max(32, height // 10)

    for index, y0 in enumerate(
        range(0, height, band_height)
    ):
        y1 = min(height, y0 + band_height)
        fill = palette[index % len(palette)]

        draw.rectangle(
            [0, y0, width, y1],
            fill=fill,
        )

        center_y = y0 + band_height // 2

        for x in range(24, width, 72):
            radius = max(8, band_height // 4)

            draw.polygon(
                _diamond_points(
                    x,
                    center_y,
                    radius,
                ),
                outline=palette[(index + 1) % len(palette)],
                width=3,
            )


def _draw_vertical_bands(
    draw: ImageDraw.ImageDraw,
    palette: Palette,
    width: int,
    height: int,
    rng: random.Random,
) -> None:
    del rng

    band_width = max(32, width // 10)

    for index, x0 in enumerate(
        range(0, width, band_width)
    ):
        x1 = min(width, x0 + band_width)
        fill = palette[index % len(palette)]

        draw.rectangle(
            [x0, 0, x1, height],
            fill=fill,
        )

        center_x = x0 + band_width // 2

        for y in range(24, height, 72):
            radius = max(8, band_width // 4)

            draw.polygon(
                _diamond_points(
                    center_x,
                    y,
                    radius,
                ),
                outline=palette[(index + 1) % len(palette)],
                width=3,
            )


def _draw_diamonds_central(
    draw: ImageDraw.ImageDraw,
    palette: Palette,
    width: int,
    height: int,
    rng: random.Random,
) -> None:
    del rng

    draw.rectangle(
        [0, 0, width, height],
        fill=palette[-1],
    )

    center_x = width // 2
    center_y = height // 2

    radii = [
        int(min(width, height) * fraction)
        for fraction in (0.43, 0.33, 0.23, 0.13, 0.055)
    ]

    for index, radius in enumerate(radii):
        draw.polygon(
            _diamond_points(
                center_x,
                center_y,
                radius,
            ),
            fill=palette[index % len(palette)],
            outline=palette[(index + 1) % len(palette)],
        )

    side_radius = max(24, min(width, height) // 12)

    for offset_x, offset_y in (
        (-width // 3, 0),
        (width // 3, 0),
        (0, -height // 3),
        (0, height // 3),
    ):
        draw.polygon(
            _diamond_points(
                center_x + offset_x,
                center_y + offset_y,
                side_radius,
            ),
            fill=palette[1 % len(palette)],
            outline=palette[0],
            width=4,
        )


def _draw_grecas_modular(
    draw: ImageDraw.ImageDraw,
    palette: Palette,
    width: int,
    height: int,
    rng: random.Random,
) -> None:
    del rng

    draw.rectangle(
        [0, 0, width, height],
        fill=palette[2 % len(palette)],
    )

    cell = max(72, min(width, height) // 6)

    for row, y0 in enumerate(range(0, height, cell)):
        for column, x0 in enumerate(range(0, width, cell)):
            color = palette[
                (row + column) % len(palette)
            ]

            points = [
                (x0 + 10, y0 + 18),
                (x0 + cell - 18, y0 + 18),
                (x0 + cell - 18, y0 + cell // 2),
                (x0 + cell // 2, y0 + cell // 2),
                (x0 + cell // 2, y0 + cell - 16),
                (x0 + cell - 8, y0 + cell - 16),
            ]

            draw.line(
                points,
                fill=color,
                width=max(5, cell // 12),
                joint="curve",
            )


def _draw_grid_reticular(
    draw: ImageDraw.ImageDraw,
    palette: Palette,
    width: int,
    height: int,
    rng: random.Random,
) -> None:
    del rng

    draw.rectangle(
        [0, 0, width, height],
        fill=palette[-1],
    )

    cell = max(56, min(width, height) // 8)

    for x in range(0, width + 1, cell):
        draw.line(
            [(x, 0), (x, height)],
            fill=palette[0],
            width=4,
        )

    for y in range(0, height + 1, cell):
        draw.line(
            [(0, y), (width, y)],
            fill=palette[0],
            width=4,
        )

    for row, y0 in enumerate(range(0, height, cell)):
        for column, x0 in enumerate(range(0, width, cell)):
            inset = max(10, cell // 5)
            color = palette[
                (row + column + 1) % len(palette)
            ]

            if (row + column) % 2 == 0:
                draw.ellipse(
                    [
                        x0 + inset,
                        y0 + inset,
                        min(width, x0 + cell - inset),
                        min(height, y0 + cell - inset),
                    ],
                    fill=color,
                )
            else:
                draw.rectangle(
                    [
                        x0 + inset,
                        y0 + inset,
                        min(width, x0 + cell - inset),
                        min(height, y0 + cell - inset),
                    ],
                    fill=color,
                )


def _draw_mixed_asymmetric(
    draw: ImageDraw.ImageDraw,
    palette: Palette,
    width: int,
    height: int,
    rng: random.Random,
) -> None:
    draw.rectangle(
        [0, 0, width, height],
        fill=palette[-1],
    )

    draw.rectangle(
        [0, 0, width // 5, height],
        fill=palette[0],
    )

    draw.rectangle(
        [0, int(height * 0.72), width, height],
        fill=palette[1 % len(palette)],
    )

    for index in range(24):
        center_x = rng.randint(
            width // 4,
            width - 25,
        )
        center_y = rng.randint(
            20,
            int(height * 0.78),
        )
        radius = rng.randint(
            max(12, width // 40),
            max(24, width // 13),
        )
        color = palette[index % len(palette)]

        shape = rng.choice(
            ("diamond", "rectangle", "line")
        )

        if shape == "diamond":
            draw.polygon(
                _diamond_points(
                    center_x,
                    center_y,
                    radius,
                ),
                outline=color,
                width=4,
            )

        elif shape == "rectangle":
            draw.rectangle(
                [
                    center_x - radius,
                    center_y - radius // 2,
                    center_x + radius,
                    center_y + radius // 2,
                ],
                outline=color,
                width=4,
            )

        else:
            draw.line(
                [
                    (
                        center_x - radius,
                        center_y + radius,
                    ),
                    (
                        center_x + radius,
                        center_y - radius,
                    ),
                ],
                fill=color,
                width=5,
            )


def _draw_chevrons_diagonal(
    draw: ImageDraw.ImageDraw,
    palette: Palette,
    width: int,
    height: int,
    rng: random.Random,
) -> None:
    del rng

    draw.rectangle(
        [0, 0, width, height],
        fill=palette[-1],
    )

    row_step = max(48, height // 9)
    chevron_width = max(64, width // 7)

    for row_index, y in enumerate(
        range(-row_step, height + row_step, row_step)
    ):
        offset = (
            -chevron_width // 2
            if row_index % 2
            else 0
        )

        for column_index, x in enumerate(
            range(
                offset - chevron_width,
                width + chevron_width,
                chevron_width,
            )
        ):
            color = palette[
                (row_index + column_index) % len(palette)
            ]

            draw.line(
                [
                    (x, y),
                    (
                        x + chevron_width // 2,
                        y + row_step // 2,
                    ),
                    (
                        x + chevron_width,
                        y,
                    ),
                ],
                fill=color,
                width=max(6, row_step // 7),
                joint="curve",
            )


def _draw_circles_concentric(
    draw: ImageDraw.ImageDraw,
    palette: Palette,
    width: int,
    height: int,
    rng: random.Random,
) -> None:
    del rng

    draw.rectangle(
        [0, 0, width, height],
        fill=palette[-1],
    )

    center_x = width // 2
    center_y = height // 2
    max_radius = int(min(width, height) * 0.44)
    ring_step = max(24, max_radius // 7)

    for index, radius in enumerate(
        range(max_radius, 10, -ring_step)
    ):
        color = palette[index % len(palette)]

        draw.ellipse(
            [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ],
            outline=color,
            width=max(8, ring_step // 3),
        )

    spoke_radius = max_radius

    for dx, dy in (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ):
        length = (
            spoke_radius
            if dx == 0 or dy == 0
            else int(spoke_radius * 0.70)
        )

        draw.line(
            [
                (center_x, center_y),
                (
                    center_x + dx * length,
                    center_y + dy * length,
                ),
            ],
            fill=palette[0],
            width=4,
        )

    center_radius = max(12, ring_step // 2)

    draw.ellipse(
        [
            center_x - center_radius,
            center_y - center_radius,
            center_x + center_radius,
            center_y + center_radius,
        ],
        fill=palette[1 % len(palette)],
    )


RENDERERS: dict[str, DrawFunction] = {
    "bands_horizontal": _draw_horizontal_bands,
    "bands_vertical": _draw_vertical_bands,
    "diamonds_central": _draw_diamonds_central,
    "grecas_modular": _draw_grecas_modular,
    "grid_reticular": _draw_grid_reticular,
    "mixed_asymmetric": _draw_mixed_asymmetric,
    "chevrons_diagonal": _draw_chevrons_diagonal,
    "circles_concentric": _draw_circles_concentric,
}


def render_pattern(
    pattern_id: str,
    raw_palette: Sequence[Sequence[int]],
    seed: int,
    width: int = 512,
    height: int = 512,
) -> Image.Image:
    """Renderiza un patrón mediante un generador local determinista."""

    if width <= 0 or height <= 0:
        raise ValueError(
            "El ancho y el alto deben ser positivos."
        )

    if pattern_id not in RENDERERS:
        raise ValueError(
            f"Patrón desconocido: {pattern_id}"
        )

    palette = normalize_palette(raw_palette)
    rng = random.Random(int(seed))

    image = Image.new(
        "RGB",
        (int(width), int(height)),
        color=palette[-1],
    )

    draw = ImageDraw.Draw(image)

    RENDERERS[pattern_id](
        draw,
        palette,
        int(width),
        int(height),
        rng,
    )

    return image


def save_contact_sheet(
    image_paths: Sequence[Path],
    labels: Sequence[str],
    output_path: Path,
    columns: int = 4,
) -> None:
    """Construye una lámina de contacto para inspección visual."""

    if len(image_paths) != len(labels):
        raise ValueError(
            "Cada imagen debe tener una etiqueta."
        )

    if not image_paths:
        raise ValueError(
            "Se requiere al menos una imagen."
        )

    if columns <= 0:
        raise ValueError(
            "El número de columnas debe ser positivo."
        )

    tile_width = 256
    tile_height = 292
    label_height = 34

    rows = (
        len(image_paths) + columns - 1
    ) // columns

    sheet = Image.new(
        "RGB",
        (
            columns * tile_width,
            rows * tile_height,
        ),
        color=(245, 245, 245),
    )

    sheet_draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (path, label) in enumerate(
        zip(image_paths, labels, strict=True)
    ):
        with Image.open(path) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail(
                (
                    tile_width - 16,
                    tile_height - label_height - 16,
                )
            )

        column = index % columns
        row = index // columns

        origin_x = column * tile_width
        origin_y = row * tile_height

        image_x = (
            origin_x
            + (tile_width - thumbnail.width) // 2
        )
        image_y = origin_y + 8

        sheet.paste(
            thumbnail,
            (image_x, image_y),
        )

        sheet_draw.rectangle(
            [
                origin_x,
                origin_y,
                origin_x + tile_width - 1,
                origin_y + tile_height - 1,
            ],
            outline=(180, 180, 180),
            width=1,
        )

        sheet_draw.text(
            (
                origin_x + 8,
                origin_y + tile_height - label_height + 8,
            ),
            label,
            fill=(20, 20, 20),
            font=font,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        format="PNG",
    )