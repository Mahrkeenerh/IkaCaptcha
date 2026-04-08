"""
Synthetic Ikariam-style captcha generator for visual comparison and tweaking.

Run from the repo root:
    python scripts/generate_captcha.py
Outputs 9 sample images to generated_samples/ for side-by-side comparison.
"""

import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Anchor relative paths to the repo root so the script works from any CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "generated_samples"

WIDTH, HEIGHT = 400, 76
CHARSET = "abcdefghjklmnpqrstuvwxy23457"  # 28 chars matching real captchas
# Characters that are easily confused — oversampled in label generation
CONFUSABLES = "eqchbt7325"  # actual confusable pairs from error analysis

def random_bg_color():
    """Random pastel/light background color."""
    return (random.randint(180, 245), random.randint(180, 245), random.randint(180, 245))


def random_char_color():
    """Random muted/dark character color."""
    return (random.randint(20, 160), random.randint(20, 160), random.randint(20, 160))


def random_noise_color():
    """Random noise color — full spectrum."""
    return (random.randint(0, 230), random.randint(0, 230), random.randint(0, 230))

# All available Latin-script fonts for maximum variety
FONT_PATHS = [
    # DejaVu family
    "fonts/system/dejavu/DejaVuSans-Bold.ttf",
    "fonts/system/dejavu/DejaVuSans.ttf",
    "fonts/system/dejavu/DejaVuSans-Oblique.ttf",
    "fonts/system/dejavu/DejaVuSans-BoldOblique.ttf",
    "fonts/system/dejavu/DejaVuSerif-Bold.ttf",
    "fonts/system/dejavu/DejaVuSerif.ttf",
    "fonts/system/dejavu/DejaVuSerif-Italic.ttf",
    "fonts/system/dejavu/DejaVuSerif-BoldItalic.ttf",
    "fonts/system/dejavu/DejaVuSansMono-Bold.ttf",
    "fonts/system/dejavu/DejaVuSansMono.ttf",
    "fonts/system/dejavu/DejaVuSansCondensed-Bold.ttf",
    "fonts/system/dejavu/DejaVuSansCondensed.ttf",
    # FreeFont family
    "fonts/system/freefont/FreeSerif.ttf",
    "fonts/system/freefont/FreeSerifBold.ttf",
    "fonts/system/freefont/FreeSerifItalic.ttf",
    "fonts/system/freefont/FreeSerifBoldItalic.ttf",
    "fonts/system/freefont/FreeSansBold.ttf",
    "fonts/system/freefont/FreeSans.ttf",
    "fonts/system/freefont/FreeSansOblique.ttf",
    "fonts/system/freefont/FreeMonoBold.ttf",
    "fonts/system/freefont/FreeMono.ttf",
    # MS core fonts
    "fonts/system/msttcorefonts/Arial_Bold.ttf",
    "fonts/system/msttcorefonts/Arial.ttf",
    "fonts/system/msttcorefonts/Arial_Italic.ttf",
    "fonts/system/msttcorefonts/Arial_Black.ttf",
    "fonts/system/msttcorefonts/Comic_Sans_MS_Bold.ttf",
    "fonts/system/msttcorefonts/Comic_Sans_MS.ttf",
    "fonts/system/msttcorefonts/Courier_New_Bold.ttf",
    "fonts/system/msttcorefonts/Courier_New.ttf",
    "fonts/system/msttcorefonts/Georgia_Bold.ttf",
    "fonts/system/msttcorefonts/Georgia.ttf",
    "fonts/system/msttcorefonts/Georgia_Italic.ttf",
    "fonts/system/msttcorefonts/Impact.ttf",
    "fonts/system/msttcorefonts/Times_New_Roman_Bold.ttf",
    "fonts/system/msttcorefonts/Times_New_Roman.ttf",
    "fonts/system/msttcorefonts/Trebuchet_MS_Bold.ttf",
    "fonts/system/msttcorefonts/Trebuchet_MS.ttf",
    "fonts/system/msttcorefonts/Verdana_Bold.ttf",
    "fonts/system/msttcorefonts/Verdana.ttf",
    # Liberation family
    "fonts/system/liberation/LiberationSans-Bold.ttf",
    "fonts/system/liberation/LiberationSans-Regular.ttf",
    "fonts/system/liberation/LiberationSerif-Bold.ttf",
    "fonts/system/liberation/LiberationSerif-Regular.ttf",
    "fonts/system/liberation/LiberationSansNarrow-Bold.ttf",
    "fonts/system/liberation/LiberationSansNarrow-Regular.ttf",
    "fonts/system/liberation/LiberationMono-Bold.ttf",
    # Lato family
    "fonts/system/lato/Lato-Bold.ttf",
    "fonts/system/lato/Lato-Regular.ttf",
    "fonts/system/lato/Lato-Black.ttf",
    "fonts/system/lato/Lato-Light.ttf",
    "fonts/system/lato/Lato-Heavy.ttf",
    "fonts/system/lato/Lato-Italic.ttf",
    # Noto (Latin-only ones)
    "fonts/system/noto/NotoSans-Bold.ttf",
    "fonts/system/noto/NotoSans-Regular.ttf",
    "fonts/system/noto/NotoSans-Italic.ttf",
    "fonts/system/noto/NotoSans-BoldItalic.ttf",
    "fonts/system/noto/NotoSerif-Bold.ttf",
    "fonts/system/noto/NotoSerif-Regular.ttf",
    "fonts/system/noto/NotoSerif-Italic.ttf",
    "fonts/system/noto/NotoSansDisplay-Bold.ttf",
    "fonts/system/noto/NotoSansDisplay-Regular.ttf",
    "fonts/system/noto/NotoSerifDisplay-Bold.ttf",
    "fonts/system/noto/NotoSansMono-Bold.ttf",
    "fonts/system/noto/NotoSansMono-Regular.ttf",
    # Ubuntu family
    "fonts/system/ubuntu/Ubuntu-B.ttf",
    "fonts/system/ubuntu/Ubuntu-R.ttf",
    "fonts/system/ubuntu/Ubuntu-L.ttf",
    "fonts/system/ubuntu/Ubuntu-M.ttf",
    "fonts/system/ubuntu/Ubuntu-BI.ttf",
    "fonts/system/ubuntu/Ubuntu-C.ttf",
    "fonts/system/ubuntu/UbuntuMono-B.ttf",
    "fonts/system/ubuntu/UbuntuMono-R.ttf",
    # RobotoSlab
    "fonts/system/roboto-slab/RobotoSlab-Bold.ttf",
    "fonts/system/roboto-slab/RobotoSlab-Regular.ttf",
    "fonts/system/roboto-slab/RobotoSlab-Light.ttf",
    # OpenSymbol / LibreOffice
    "fonts/system/libreoffice/opens___.ttf",
    # Downloaded decorative/display fonts
    "fonts/AbrilFatface-Regular.ttf",
    "fonts/AlfaSlabOne-Regular.ttf",
    "fonts/Anton-Regular.ttf",
    "fonts/ArchivoBlack-Regular.ttf",
    "fonts/Bangers-Regular.ttf",
    "fonts/Bungee-Regular.ttf",
    "fonts/Caveat-Bold.ttf",
    "fonts/Cinzel-Bold.ttf",
    "fonts/FredokaOne-Regular.ttf",
    "fonts/Orbitron-Bold.ttf",
    "fonts/Oswald-Bold.ttf",
    "fonts/Pacifico-Regular.ttf",
    "fonts/PermanentMarker-Regular.ttf",
    "fonts/PlayfairDisplay-Bold.ttf",
    "fonts/PressStart2P-Regular.ttf",
    "fonts/Righteous-Regular.ttf",
    "fonts/RussoOne-Regular.ttf",
    "fonts/Satisfy-Regular.ttf",
    "fonts/SpecialElite-Regular.ttf",
    # Handwritten / hand-drawn
    "fonts/IndieFlower-Regular.ttf",
    "fonts/PatrickHand-Regular.ttf",
    "fonts/Kalam-Bold.ttf",
    "fonts/CoveredByYourGrace.ttf",
    "fonts/GloriaHallelujah.ttf",
    "fonts/Handlee-Regular.ttf",
    "fonts/ShadowsIntoLight.ttf",
    "fonts/Neucha.ttf",
    # Slab serif / Roman / flat-top
    "fonts/RobotoSlab-Variable.ttf",
    "fonts/Arvo-Bold.ttf",
    "fonts/Arvo-Regular.ttf",
    "fonts/Crete-Round.ttf",
    "fonts/Rokkitt-Variable.ttf",
    "fonts/Tinos-Bold.ttf",
    "fonts/Tinos-Regular.ttf",
    "fonts/Ultra-Regular.ttf",
    "fonts/Patua-One.ttf",
    "fonts/Vollkorn-Variable.ttf",
    "fonts/CormorantGaramond-Bold.ttf",
    "fonts/Spectral-Bold.ttf",
    # Script / calligraphic / swash (decorative tails/spirals)
    "fonts/GreatVibes-Regular.ttf",
    "fonts/DancingScript-Bold.ttf",
    "fonts/Courgette-Regular.ttf",
    "fonts/Cookie-Regular.ttf",
    "fonts/Sacramento-Regular.ttf",
    "fonts/Tangerine-Bold.ttf",
    "fonts/Kaushan-Script.ttf",
    "fonts/Yellowtail-Regular.ttf",
]

# Resolve all font paths to absolute (anchored to repo root) so the script
# works regardless of the caller's current working directory.
FONT_PATHS = [str(REPO_ROOT / p) for p in FONT_PATHS]


def load_all_fonts(size: int) -> list:
    """Load all available fonts at given size."""
    fonts = []
    for path in FONT_PATHS:
        try:
            fonts.append(ImageFont.truetype(path, size))
        except (OSError, IOError):
            pass
    if not fonts:
        fonts.append(ImageFont.load_default())
    return fonts


# Pre-load font objects at a few sizes for per-character size variation
_font_cache: dict[int, list] = {}


def get_random_font(size: int) -> ImageFont.FreeTypeFont:
    """Get a random font at the given size."""
    if size not in _font_cache:
        _font_cache[size] = load_all_fonts(size)
    return random.choice(_font_cache[size])


def render_char(char: str, font: ImageFont.FreeTypeFont, color: tuple, rotation: int, style: str) -> Image.Image:
    """Render a single character with optional 3D/shadow effect, rotated."""
    bbox = font.getbbox(char)
    pad = 12
    cw = bbox[2] - bbox[0] + pad * 2
    ch = bbox[3] - bbox[1] + pad * 2

    char_img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    char_draw = ImageDraw.Draw(char_img)
    ox, oy = -bbox[0] + pad, -bbox[1] + pad

    if style == "3d":
        # Shadow/3D offset
        shadow_color = tuple(max(0, c - 60) for c in color)
        for dx, dy in [(2, 2), (1, 2), (2, 1)]:
            char_draw.text((ox + dx, oy + dy), char, fill=shadow_color + (180,), font=font)
        # Outline
        outline_color = tuple(min(255, c + 80) for c in color)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                char_draw.text((ox + dx, oy + dy), char, fill=outline_color + (200,), font=font)
        char_draw.text((ox, oy), char, fill=color + (255,), font=font)

    elif style == "outline":
        # Just outline, no fill shadow
        outline_color = tuple(min(255, c + 60) for c in color)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                char_draw.text((ox + dx, oy + dy), char, fill=outline_color + (220,), font=font)
        char_draw.text((ox, oy), char, fill=color + (255,), font=font)

    else:  # "plain"
        char_draw.text((ox, oy), char, fill=color + (255,), font=font)

    # Spatial distortion — randomly pick mesh or elastic (or none)
    distortion = random.choices(["mesh", "elastic", "none"], weights=[0.4, 0.3, 0.3])[0]

    if distortion == "mesh":
        char_img = _mesh_distort(char_img)
    elif distortion == "elastic":
        char_img = _elastic_distort(char_img)

    rotated = char_img.rotate(rotation, expand=True, resample=Image.BICUBIC)
    return rotated


def _mesh_distort(img, grid=3, strength=0.25):
    """Mesh grid distortion: divide into grid, jitter each control point."""
    w, h = img.size
    # Build grid of control points
    xs = [int(w * i / (grid - 1)) for i in range(grid)]
    ys = [int(h * i / (grid - 1)) for i in range(grid)]

    # Jitter each point
    jittered = {}
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            # Don't jitter too much at edges to avoid clipping
            mx = int(w * strength / (grid - 1))
            my = int(h * strength / (grid - 1))
            jx = x + random.randint(-mx, mx)
            jy = y + random.randint(-my, my)
            jittered[(ix, iy)] = (jx, jy)

    # Build mesh quads: each cell maps from jittered quad to regular rect
    meshes = []
    for iy in range(grid - 1):
        for ix in range(grid - 1):
            # Target rectangle (where pixels end up)
            rect = (xs[ix], ys[iy], xs[ix + 1], ys[iy + 1])
            # Source quad (where pixels come from) — the jittered corners
            tl = jittered[(ix, iy)]
            tr = jittered[(ix + 1, iy)]
            br = jittered[(ix + 1, iy + 1)]
            bl = jittered[(ix, iy + 1)]
            quad = (tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1])
            meshes.append((rect, quad))

    return img.transform(img.size, Image.MESH, meshes, resample=Image.BICUBIC)


def _elastic_distort(img, alpha=6.0, sigma=3.0):
    """Elastic distortion (Simard-style): smooth random displacement field."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    w, h = img.size
    # Random displacement fields
    dx = gaussian_filter(np.random.randn(h, w) * alpha, sigma)
    dy = gaussian_filter(np.random.randn(h, w) * alpha, sigma)

    # Build pixel coordinate maps
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    map_x = (x_coords + dx).astype(np.float32)
    map_y = (y_coords + dy).astype(np.float32)

    # Apply per channel
    img_arr = np.array(img)
    from scipy.ndimage import map_coordinates
    if img_arr.ndim == 3:
        channels = []
        for c in range(img_arr.shape[2]):
            channels.append(map_coordinates(img_arr[:, :, c], [map_y, map_x], order=1, mode='constant', cval=0))
        result = np.stack(channels, axis=-1).astype(np.uint8)
    else:
        result = map_coordinates(img_arr, [map_y, map_x], order=1, mode='constant', cval=0).astype(np.uint8)

    return Image.fromarray(result, img.mode)


def draw_noise_lines(draw, count: int):
    """Draw thin colored lines across the image."""
    for _ in range(count):
        color = random_noise_color()
        x1 = random.randint(-50, WIDTH + 50)
        y1 = random.randint(-30, HEIGHT + 30)
        x2 = random.randint(-50, WIDTH + 50)
        y2 = random.randint(-30, HEIGHT + 30)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=random.choice([1, 1, 1, 2]))


def draw_noise_ellipses(img, count: int):
    """Draw ellipses with random fill colors and a different outline color."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for _ in range(count):
        cx = random.randint(-20, WIDTH + 40)
        cy = random.randint(-20, HEIGHT + 40)
        # Smaller ellipses: radius 8-50 (was 15-80)
        rx = random.randint(8, 50)
        ry = random.randint(8, 40)
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]

        # Random fill color (fully random, not tied to bg)
        if random.random() < 0.5:
            fill = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(80, 160),
            )
            overlay_draw.ellipse(bbox, fill=fill)

        # Outline — always present, different random color
        outline_color = (
            random.randint(20, 220),
            random.randint(20, 220),
            random.randint(20, 220),
        )
        outline_width = random.choice([1, 1, 2])
        overlay_draw.ellipse(bbox, outline=outline_color, width=outline_width)

    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"), (0, 0))


def generate_captcha(label: str | None = None) -> tuple[Image.Image, str]:
    """Generate a single captcha image. Returns (image, label)."""
    if label is None:
        length = random.randint(4, 8)
        chars = []
        for _ in range(length):
            # 50% chance to pick from confusable characters
            if random.random() < 0.5:
                chars.append(random.choice(CONFUSABLES))
            else:
                chars.append(random.choice(CHARSET))
        label = "".join(chars)

    bg_color = random_bg_color()
    img = Image.new("RGB", (WIDTH, HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)

    # Lines behind characters
    draw_noise_lines(draw, count=random.randint(2, 12))

    # Ellipses behind characters
    draw_noise_ellipses(img, count=random.randint(3, 6))

    # Draw characters
    # Characters spread across left ~60-70% of image (randomized)
    char_area_pct = random.uniform(0.58, 0.72)
    char_area_width = int(WIDTH * char_area_pct)
    spacing = char_area_width // (len(label) + 1)

    for i, char in enumerate(label):
        # Per-character random font size for variety
        font_size = random.randint(24, 48)
        font = get_random_font(font_size)
        color = random_char_color()
        # Wide rotation variety
        rotation = random.randint(-40, 40)
        x = 10 + spacing * (i + 1) - spacing // 2 + random.randint(-5, 5)
        y_offset = random.randint(-8, 18)

        # Randomly show as upper or lowercase visually (label stays lowercase)
        display_char = char.upper() if random.random() < 0.5 else char

        # Mix of styles: plain, outline, 3d
        style = random.choices(["plain", "outline", "3d"], weights=[0.4, 0.3, 0.3])[0]

        char_img = render_char(display_char, font, color, rotation, style)
        img.paste(char_img, (x - char_img.width // 2, y_offset), char_img)

    # Noise on top of characters
    draw = ImageDraw.Draw(img)
    draw_noise_lines(draw, count=random.randint(2, 12))
    draw_noise_ellipses(img, count=random.randint(3, 7))

    # Post-processing augmentation
    # Random slight blur
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    # Random brightness/contrast shift
    if random.random() < 0.3:
        from PIL import ImageEnhance
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.3:
        from PIL import ImageEnhance
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

    return img, label


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    for i in range(9):
        img, label = generate_captcha()
        path = OUTPUT_DIR / f"gen_{i+1}_{label}.png"
        img.save(path)
        print(f"Saved {path}")

    print(f"\nGenerated 9 samples in {OUTPUT_DIR}/")
    print("Compare with samples/ to see how close they are.")
