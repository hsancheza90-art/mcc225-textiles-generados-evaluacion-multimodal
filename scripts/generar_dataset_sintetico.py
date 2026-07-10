from pathlib import Path
import csv
from PIL import Image, ImageDraw
import random

# ============================================================
# Configuración general
# ============================================================

SEED = 225
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = BASE_DIR / "data" / "images"
MANIFEST_PATH = BASE_DIR / "data" / "manifest_textiles_generados.csv"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 512
HEIGHT = 512

PALETTES = {
    "rojo_negro_crema": [(120, 35, 30), (30, 25, 25), (225, 205, 165)],
    "azul_crema_marron": [(35, 70, 120), (230, 210, 170), (95, 60, 40)],
    "ocre_marron_negro": [(180, 120, 55), (95, 55, 30), (25, 25, 25)],
    "blanco_negro_gris": [(235, 235, 225), (30, 30, 30), (130, 130, 130)],
    "multicolor_tierra": [(150, 55, 45), (220, 160, 65), (45, 95, 80), (35, 45, 100)]
}

CONFIGS = [
    ("bandas_horizontales", "horizontal", "bandas geometricas", "repetitiva"),
    ("bandas_verticales", "vertical", "bandas geometricas", "repetitiva"),
    ("rombos", "central", "rombos", "bilateral"),
    ("grecas", "modular", "grecas escalonadas", "repetitiva"),
    ("reticula", "reticular", "cuadricula geometrica", "repetitiva"),
    ("mixto_ambiguo", "mixta", "motivos geometricos combinados", "asimetrica")
]


# ============================================================
# Funciones de dibujo
# ============================================================

def draw_horizontal_bands(draw, palette):
    band_h = HEIGHT // 12
    for i in range(12):
        color = palette[i % len(palette)]
        y0 = i * band_h
        y1 = HEIGHT if i == 11 else (i + 1) * band_h
        draw.rectangle([0, y0, WIDTH, y1], fill=color)

    for y in range(40, HEIGHT, 80):
        for x in range(30, WIDTH, 80):
            draw.rectangle([x, y, x + 35, y + 20], outline=palette[-1], width=3)


def draw_vertical_bands(draw, palette):
    band_w = WIDTH // 12
    for i in range(12):
        color = palette[i % len(palette)]
        x0 = i * band_w
        x1 = WIDTH if i == 11 else (i + 1) * band_w
        draw.rectangle([x0, 0, x1, HEIGHT], fill=color)

    for x in range(40, WIDTH, 80):
        for y in range(30, HEIGHT, 80):
            draw.rectangle([x, y, x + 20, y + 35], outline=palette[-1], width=3)


def draw_diamonds(draw, palette):
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=palette[2 % len(palette)])
    step = 96
    size = 42
    for y in range(64, HEIGHT, step):
        for x in range(64, WIDTH, step):
            color = palette[(x + y) % len(palette)]
            points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
            draw.polygon(points, fill=color, outline=palette[0])


def draw_stepped_grecas(draw, palette):
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=palette[1 % len(palette)])
    cell = 64
    for y in range(0, HEIGHT, cell):
        for x in range(0, WIDTH, cell):
            color = palette[(x // cell + y // cell) % len(palette)]
            draw.rectangle([x + 8, y + 8, x + 48, y + 20], fill=color)
            draw.rectangle([x + 36, y + 8, x + 48, y + 48], fill=color)
            draw.rectangle([x + 20, y + 36, x + 48, y + 48], fill=color)


def draw_grid(draw, palette):
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=palette[2 % len(palette)])
    cell = 64
    for y in range(0, HEIGHT, cell):
        for x in range(0, WIDTH, cell):
            color = palette[(x // cell + y // cell) % len(palette)]
            draw.rectangle([x + 6, y + 6, x + cell - 6, y + cell - 6], outline=color, width=4)
            if (x + y) // cell % 2 == 0:
                draw.ellipse([x + 22, y + 22, x + 42, y + 42], fill=color)


def draw_mixed(draw, palette):
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=palette[2 % len(palette)])
    draw_horizontal_bands(draw, palette)
    for _ in range(20):
        x = random.randint(30, WIDTH - 30)
        y = random.randint(30, HEIGHT - 30)
        size = random.randint(18, 42)
        color = random.choice(palette)
        points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
        draw.polygon(points, outline=color, width=3)


def create_image(pattern, palette):
    image = Image.new("RGB", (WIDTH, HEIGHT), color=palette[0])
    draw = ImageDraw.Draw(image)

    if pattern == "bandas_horizontales":
        draw_horizontal_bands(draw, palette)
    elif pattern == "bandas_verticales":
        draw_vertical_bands(draw, palette)
    elif pattern == "rombos":
        draw_diamonds(draw, palette)
    elif pattern == "grecas":
        draw_stepped_grecas(draw, palette)
    elif pattern == "reticula":
        draw_grid(draw, palette)
    else:
        draw_mixed(draw, palette)

    return image


# ============================================================
# Construcción de captions
# ============================================================

def build_captions(pattern, palette_name, composicion, motivo, simetria, ambiguity):
    palette_text = palette_name.replace("_", ", ")

    caption_1 = (
        f"Patron textil generado con composicion {composicion}, "
        f"motivos de {motivo} y paleta {palette_text}."
    )

    caption_2 = (
        f"Imagen generada de inspiracion textil con {motivo}, "
        f"organizacion {composicion} y simetria {simetria}."
    )

    caption_3 = (
        f"Diseno geometrico generado con paleta {palette_text}, "
        f"estructura {composicion} y repeticion visual."
    )

    caption_4 = (
        f"Patron sintetico con elementos de {motivo}, "
        f"distribucion {composicion} y contraste cromatico visible."
    )

    caption_5 = (
        f"Imagen de patron ornamental generado con rasgos geometricos, "
        f"simetria {simetria} y nivel de ambiguedad {ambiguity}."
    )

    return caption_1, caption_2, caption_3, caption_4, caption_5


# ============================================================
# Generación del dataset
# ============================================================

def main():
    rows = []
    total = 40

    palette_names = list(PALETTES.keys())

    for idx in range(1, total + 1):
        pattern, composicion, motivo, simetria = CONFIGS[(idx - 1) % len(CONFIGS)]
        palette_name = palette_names[(idx - 1) % len(palette_names)]
        palette = PALETTES[palette_name]

        if pattern == "mixto_ambiguo":
            ambiguity = "alto"
        elif pattern in ["rombos", "grecas"]:
            ambiguity = "medio"
        else:
            ambiguity = "bajo"

        image_id = f"T{idx:03d}"
        image_path = f"data/images/{image_id}.png"

        image = create_image(pattern, palette)
        image.save(IMAGE_DIR / f"{image_id}.png")

        captions = build_captions(
            pattern=pattern,
            palette_name=palette_name,
            composicion=composicion,
            motivo=motivo,
            simetria=simetria,
            ambiguity=ambiguity
        )

        rows.append({
            "image_id": image_id,
            "image_path": image_path,
            "caption_1": captions[0],
            "caption_2": captions[1],
            "caption_3": captions[2],
            "caption_4": captions[3],
            "caption_5": captions[4],
            "paleta": palette_name,
            "composicion": composicion,
            "motivo": motivo,
            "simetria": simetria,
            "nivel_ambiguedad": ambiguity,
            "observacion": "Imagen generada por script para evaluacion academica controlada"
        })

    fieldnames = [
        "image_id",
        "image_path",
        "caption_1",
        "caption_2",
        "caption_3",
        "caption_4",
        "caption_5",
        "paleta",
        "composicion",
        "motivo",
        "simetria",
        "nivel_ambiguedad",
        "observacion"
    ]

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset generado correctamente.")
    print(f"Imagenes generadas: {total}")
    print(f"Directorio de imagenes: {IMAGE_DIR}")
    print(f"Manifiesto: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
