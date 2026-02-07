#!/usr/bin/env python3
"""
Phase 4: Disparity to Elevation
Convert the filtered disparity map into a relative-elevation surface.

Simplified approach — no water detection or clamping:
1. Load the filtered disparity from Phase 3
2. Light smoothing to reduce noise
3. Normalize to 16-bit elevation range

Input:  Phase 3 disparity_filtered.tif (uint8)
Output: elevation_16bit.tif (uint16), + JPG previews
"""

from PIL import Image
import cv2
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE3_DIR = os.path.join(BASE_DIR, 'intermediate-processing', 'phase3')
OUTPUT_DIR = os.path.join(BASE_DIR, 'intermediate-processing', 'phase4')


def save_with_thumbnail(arr, tif_path, jpg_path, thumb_width=1000):
    """Save as TIF and a JPG thumbnail."""
    Image.fromarray(arr).save(tif_path)
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        a_min, a_max = arr.min(), arr.max()
        a_range = a_max - a_min if a_max > a_min else 1.0
        arr_8 = ((arr - a_min) / a_range * 255).astype(np.uint8)
    elif arr.dtype == np.uint16:
        arr_8 = (arr / 256).astype(np.uint8)
    else:
        arr_8 = arr
    thumb = Image.fromarray(arr_8)
    w, h = thumb.size
    thumb = thumb.resize((thumb_width, int(thumb_width * h / w)),
                         Image.Resampling.LANCZOS)
    thumb.save(jpg_path, quality=90)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Phase 4: Disparity to Elevation ===\n")

    print("Loading filtered disparity map:")
    disparity = np.array(
        Image.open(os.path.join(PHASE3_DIR, 'disparity_filtered.tif'))
    )
    print(f"  Shape: {disparity.shape}")
    print(f"  Range: {disparity.min()} to {disparity.max()}")

    elevation = disparity.astype(np.float32)

    print("\nApplying light smoothing...")
    elevation = cv2.bilateralFilter(
        elevation, d=9, sigmaColor=50, sigmaSpace=50
    )
    print(f"  Smoothed range: {elevation.min():.1f} to {elevation.max():.1f}")

    print("\nNormalizing elevation:")
    e_min, e_max = elevation.min(), elevation.max()
    e_range = e_max - e_min if e_max > e_min else 1.0
    elevation_norm = (elevation - e_min) / e_range
    print(f"  Original range: {e_min:.1f} to {e_max:.1f}")
    print(f"  Normalized to: 0.0 to 1.0")

    save_with_thumbnail(
        elevation_norm.astype(np.float32),
        os.path.join(OUTPUT_DIR, 'elevation_float.tif'),
        os.path.join(OUTPUT_DIR, 'elevation_float.jpg')
    )
    print("\n  Saved: elevation_float.tif/.jpg")

    elevation_16 = (elevation_norm * 65535).astype(np.uint16)
    Image.fromarray(elevation_16).save(
        os.path.join(OUTPUT_DIR, 'elevation_16bit.tif')
    )
    print("  Saved: elevation_16bit.tif")

    elevation_8 = (elevation_norm * 255).astype(np.uint8)
    save_with_thumbnail(
        elevation_8,
        os.path.join(OUTPUT_DIR, 'elevation_preview.tif'),
        os.path.join(OUTPUT_DIR, 'elevation_preview.jpg')
    )
    print("  Saved: elevation_preview.tif/.jpg")

    colored = cv2.applyColorMap(elevation_8, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    save_with_thumbnail(
        colored_rgb,
        os.path.join(OUTPUT_DIR, 'elevation_colored.tif'),
        os.path.join(OUTPUT_DIR, 'elevation_colored.jpg')
    )
    print("  Saved: elevation_colored.tif/.jpg")

    print(f"\n  Elevation grid: {elevation.shape[1]}x{elevation.shape[0]}")
    print("\n=== Phase 4 complete ===")


if __name__ == '__main__':
    main()
