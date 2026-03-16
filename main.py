"""
Image Presets — Python script
Usage: python image_presets.py <path_to_image>
       python image_presets.py  (uses a generated test image)

Covers:
  - Brightness, Contrast, Exposure
  - Saturation, Vibrance, Temperature, Tint
  - Curves (lift shadows, pull highlights, S-curve)
  - Split Toning
  - Vignette
  - Sharpening
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import math
import os
import json

# ─────────────────────────────────────────────
#  CORE MATH — every adjustment as a pure fn
#  All inputs are float32 numpy arrays in [0,1]
# ─────────────────────────────────────────────

def adjust_brightness(img, value):
    """
    value in [-1, 1]
    Simple additive shift: out = in + value
    Clipped to [0, 1]
    """
    return np.clip(img + value, 0, 1)


def adjust_contrast(img, value):
    """
    value in [-1, 1]
    Scales distance from mid-grey (0.5):
      factor = (1 + value)        when value >= 0  → stretch
      factor = 1 / (1 - value)    when value <  0  → compress
    out = 0.5 + (in - 0.5) * factor
    """
    if value >= 0:
        factor = 1.0 + value
    else:
        factor = max(1.0 + value, 0.01)   # avoid zero/negative
    return np.clip(0.5 + (img - 0.5) * factor, 0, 1)


def adjust_exposure(img, stops):
    """
    stops in [-3, 3]  (like camera EV)
    out = in * 2^stops
    Logarithmic — each stop doubles/halves brightness.
    """
    return np.clip(img * (2.0 ** stops), 0, 1)


def adjust_saturation(img, value):
    """
    value in [-1, 1]
    Converts to grayscale luminance, then lerps:
      factor = 1 + value
      out = lerp(grey, colour, factor)
    Uses perceptual weights: R=0.2126, G=0.7152, B=0.0722
    """
    lum = (0.2126 * img[..., 0] +
           0.7152 * img[..., 1] +
           0.0722 * img[..., 2])[..., np.newaxis]
    factor = 1.0 + value
    return np.clip(lum + (img - lum) * factor, 0, 1)


def adjust_vibrance(img, value):
    """
    value in [-1, 1]
    Like saturation but protects already-saturated pixels.
    Per-pixel factor = 1 + value * (1 - max_channel)
    so dull pixels get boosted more than vivid ones.
    """
    max_ch = img.max(axis=2, keepdims=True)
    factor = 1.0 + value * (1.0 - max_ch)
    lum = (0.2126 * img[..., 0] +
           0.7152 * img[..., 1] +
           0.0722 * img[..., 2])[..., np.newaxis]
    return np.clip(lum + (img - lum) * factor, 0, 1)


def adjust_temperature(img, value):
    """
    value in [-1, 1]   negative = cooler (blue), positive = warmer (orange)
    Shifts R and B channels in opposite directions.
    Strength is scaled so ±1 shifts channel by ±0.15
    """
    scale = 0.15 * value
    out = img.copy()
    out[..., 0] = np.clip(img[..., 0] + scale, 0, 1)    # R up = warm
    out[..., 2] = np.clip(img[..., 2] - scale, 0, 1)    # B down = warm
    return out


def adjust_tint(img, value):
    """
    value in [-1, 1]   negative = green, positive = magenta
    Shifts G channel.
    """
    scale = 0.10 * value
    out = img.copy()
    out[..., 1] = np.clip(img[..., 1] - scale, 0, 1)    # G down = magenta
    return out


def apply_curve(img, control_points):
    """
    control_points: list of (in_val, out_val) tuples, both in [0, 1]
    Builds a LUT via monotone cubic interpolation (Fritsch-Carlson).
    Applied to all channels equally (luminance curve).
    For per-channel curves, call once per channel.
    """
    pts = sorted(control_points, key=lambda p: p[0])
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])

    # Build 256-entry LUT
    lut_x = np.linspace(0, 1, 256)
    lut_y = _monotone_cubic_interp(xs, ys, lut_x)
    lut_y = np.clip(lut_y, 0, 1)

    # Map image through LUT
    indices = np.clip((img * 255).astype(np.int32), 0, 255)
    return lut_y[indices].astype(np.float32)


def _monotone_cubic_interp(xs, ys, x_new):
    """Fritsch-Carlson monotone cubic interpolation."""
    n = len(xs)
    if n == 1:
        return np.full_like(x_new, ys[0])
    if n == 2:
        return np.interp(x_new, xs, ys)

    # Compute slopes
    delta = np.diff(ys) / np.diff(xs)
    m = np.zeros(n)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0
        else:
            m[i] = (delta[i - 1] + delta[i]) / 2

    # Monotonicity constraint
    for i in range(n - 1):
        if abs(delta[i]) < 1e-10:
            m[i] = m[i + 1] = 0
        else:
            a = m[i] / delta[i]
            b = m[i + 1] / delta[i]
            if a ** 2 + b ** 2 > 9:
                t = 3 / math.sqrt(a ** 2 + b ** 2)
                m[i] = t * a * delta[i]
                m[i + 1] = t * b * delta[i]

    # Evaluate cubic Hermite spline
    result = np.zeros_like(x_new)
    for j, xv in enumerate(x_new):
        if xv <= xs[0]:
            result[j] = ys[0]
        elif xv >= xs[-1]:
            result[j] = ys[-1]
        else:
            i = np.searchsorted(xs, xv) - 1
            i = min(i, n - 2)
            h = xs[i + 1] - xs[i]
            t = (xv - xs[i]) / h
            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 = t**3 - t**2
            result[j] = h00*ys[i] + h10*h*m[i] + h01*ys[i+1] + h11*h*m[i+1]
    return result


def apply_split_tone(img, shadow_hue, shadow_sat, highlight_hue, highlight_sat):
    """
    Tints shadows and highlights with separate hues.
    shadow_hue / highlight_hue in [0, 360]
    shadow_sat / highlight_sat in [0, 1]

    Weight of shadow tint  = 1 - luminance   (dark pixels get more)
    Weight of highlight tint = luminance      (bright pixels get more)
    Then blended into RGB.
    """
    lum = (0.2126 * img[..., 0] +
           0.7152 * img[..., 1] +
           0.0722 * img[..., 2])

    def hue_to_rgb(hue):
        h = hue / 60.0
        x = 1 - abs(h % 2 - 1)
        if   h < 1: r, g, b = 1, x, 0
        elif h < 2: r, g, b = x, 1, 0
        elif h < 3: r, g, b = 0, 1, x
        elif h < 4: r, g, b = 0, x, 1
        elif h < 5: r, g, b = x, 0, 1
        else:        r, g, b = 1, 0, x
        return np.array([r, g, b], dtype=np.float32)

    s_rgb = hue_to_rgb(shadow_hue % 360)
    h_rgb = hue_to_rgb(highlight_hue % 360)

    shadow_w    = (1 - lum)[..., np.newaxis] * shadow_sat
    highlight_w = lum[..., np.newaxis] * highlight_sat

    out = img + shadow_w * (s_rgb - img) + highlight_w * (h_rgb - img)
    return np.clip(out, 0, 1)


def apply_vignette(img, strength, feather=0.5):
    """
    strength in [-1, 1]   negative = darken edges (classic), positive = brighten
    feather  in [0, 1]    how soft the falloff is

    Distance from center is computed, then a smooth mask is applied.
    mask = 1 - strength * smoothstep(inner, outer, dist)
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    # Normalised distance: corners = 1
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2)

    inner = 1.0 - feather
    outer = 1.0 + feather
    t = np.clip((dist - inner) / (outer - inner), 0, 1)
    mask = t * t * (3 - 2 * t)           # smoothstep

    vig = 1.0 - strength * mask
    return np.clip(img * vig[..., np.newaxis], 0, 1)


def apply_sharpening(img, amount=0.5, radius=1.0):
    """
    Unsharp mask:  out = in + amount * (in - blur(in))
    amount in [0, 2]
    """
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    blur_arr = np.array(blurred).astype(np.float32) / 255.0
    sharpened = img + amount * (img - blur_arr)
    return np.clip(sharpened, 0, 1)


# ─────────────────────────────────────────────
#  PRESET PIPELINE
#  Each preset is a dict of parameter names → values.
#  apply_preset() runs them in the correct order.
# ─────────────────────────────────────────────

def apply_preset(img_float, preset):
    """
    img_float: H×W×3 float32 in [0, 1]
    preset:    dict of adjustments
    Returns processed float32 image.
    """
    p = preset
    out = img_float.copy()

    # 1. Exposure (log-scale, do before everything)
    if p.get("exposure"):
        out = adjust_exposure(out, p["exposure"])

    # 2. Brightness & Contrast
    if p.get("brightness"):
        out = adjust_brightness(out, p["brightness"])
    if p.get("contrast"):
        out = adjust_contrast(out, p["contrast"])

    # 3. Tone Curve (composite)
    if p.get("curve"):
        out = apply_curve(out, p["curve"])

    # 4. Per-channel curves (colour grading via curves)
    if p.get("curve_r"):
        out[..., 0] = apply_curve(out[..., 0][..., np.newaxis],
                                  p["curve_r"])[..., 0]
    if p.get("curve_g"):
        out[..., 1] = apply_curve(out[..., 1][..., np.newaxis],
                                  p["curve_g"])[..., 0]
    if p.get("curve_b"):
        out[..., 2] = apply_curve(out[..., 2][..., np.newaxis],
                                  p["curve_b"])[..., 0]

    # 5. White Balance
    if p.get("temperature"):
        out = adjust_temperature(out, p["temperature"])
    if p.get("tint"):
        out = adjust_tint(out, p["tint"])

    # 6. Colour
    if p.get("saturation"):
        out = adjust_saturation(out, p["saturation"])
    if p.get("vibrance"):
        out = adjust_vibrance(out, p["vibrance"])

    # 7. Split Toning
    if p.get("split_tone"):
        st = p["split_tone"]
        out = apply_split_tone(out,
                               st["shadow_hue"],   st["shadow_sat"],
                               st["highlight_hue"], st["highlight_sat"])

    # 8. Effects
    if p.get("vignette"):
        out = apply_vignette(out, p["vignette"])
    if p.get("sharpen"):
        out = apply_sharpening(out, p["sharpen"])

    return out


# ─────────────────────────────────────────────
#  LOAD IMAGE
# ─────────────────────────────────────────────

def load_image(path, max_size=1024):
    """
    Load any image file as float32 RGB in [0, 1].
    Resizes so that max(width, height) <= max_size, preserving aspect ratio.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = round(w * scale)
        new_h = round(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"Resized: ({w}, {h}) → ({new_w}, {new_h})")
    return np.array(img, dtype=np.float32) / 255.0
 

def make_test_image(size=512):
    """
    Generates a synthetic test image if no path is provided.
    Contains a gradient background + coloured circles so
    colour/tone effects are clearly visible.
    """
    h = w = size
    img = np.zeros((h, w, 3), dtype=np.float32)

    # Gradient background: warm top-left to cool bottom-right
    for i in range(h):
        for j in range(w):
            img[i, j, 0] = j / w                     # R increases left→right
            img[i, j, 1] = 0.3 + 0.4 * (i / h)      # G increases top→bottom
            img[i, j, 2] = 1.0 - (j / w)             # B decreases left→right

    # Add coloured circles
    cx, cy = w // 2, h // 2
    ys, xs = np.ogrid[:h, :w]

    circles = [
        (cx - w//4, cy - h//4, w//7, [1.0, 0.2, 0.2]),  # red
        (cx + w//4, cy - h//4, w//7, [0.2, 0.8, 0.2]),  # green
        (cx,        cy + h//4, w//7, [0.2, 0.4, 1.0]),  # blue
        (cx - w//4, cy + h//4, w//7, [1.0, 0.9, 0.1]),  # yellow
        (cx + w//4, cy + h//4, w//7, [0.9, 0.2, 0.9]),  # magenta
    ]
    for (px, py, r, colour) in circles:
        mask = ((xs - px)**2 + (ys - py)**2) < r**2
        for c, v in enumerate(colour):
            img[mask, c] = v

    # Add a bright white circle in centre
    mask_c = ((xs - cx)**2 + (ys - cy)**2) < (w//10)**2
    img[mask_c] = [1.0, 1.0, 1.0]

    return img


# ─────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────

def plot_presets(img_float, presets, cols=4):
    n = len(presets)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 4, rows * 3.6),
                             facecolor="#111111")
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for idx, (name, params) in enumerate(presets.items()):
        result = apply_preset(img_float, params)
        result_uint8 = (result * 255).clip(0, 255).astype(np.uint8)

        axes[idx].imshow(result_uint8)
        axes[idx].set_title(name, color="white",
                            fontsize=11, fontweight="bold", pad=6)
        axes[idx].axis("off")

    plt.suptitle("Image Presets", color="white",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(pad=1.2)
    # plt.savefig("presets_output.png", dpi=150,
    #             bbox_inches="tight", facecolor="#111111")
    # print("Saved → presets_output.png")
    plt.show()

def load_presets(json_path="presets.json"):
    """
    Load presets from a JSON file.
    Curve arrays [[x, y], ...] are converted to list-of-tuples for apply_curve().
    Falls back to a single empty 'Original' preset if file not found.
    """
    if not os.path.exists(json_path):
        print(f"Warning: presets file '{json_path}' not found. Using empty preset only.")
        return {"Original": {}}
 
    with open(json_path, "r") as f:
        raw = json.load(f)
 
    presets = {}
    for name, params in raw.items():
        p = {}
        for key, val in params.items():
            # Convert curve arrays [[x,y],...] → [(x,y),...]
            if key in ("curve", "curve_r", "curve_g", "curve_b") and isinstance(val, list):
                p[key] = [tuple(pt) for pt in val]
            else:
                p[key] = val
        presets[name] = p
 
    print(f"Loaded {len(presets)} presets from '{json_path}'")
    return presets

if __name__ == '__main__':
    imgs = os.listdir('images_j')
    for i in imgs:
        image_path = f'images_j/{i}'
        print(f"Loading image: {image_path}")
        img = load_image(image_path)
        json_path ="presets.json"
        PRESETS = load_presets(json_path)

        print(f"Image shape: {img.shape}  dtype: {img.dtype}")
        print(f"Applying {len(PRESETS)} presets…")
        plot_presets(img, PRESETS, cols=4)
