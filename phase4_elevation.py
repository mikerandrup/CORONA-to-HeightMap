#!/usr/bin/env python3
"""
Phase 4: Disparity to Elevation
Convert the filtered disparity map into a clean relative-elevation surface.

Updated to work with corrected Phase 3 output where the homography has been
removed and real stereo parallax is preserved. The disparity map now covers
only the cropped overlap region.

Approach:
1. Load the filtered disparity from Phase 3
2. Detect water body (Lake Nasser) from the aft crop — force to flat zero elevation
3. Inpaint small terrain gaps only (not water)
4. Smooth terrain while preserving the sharp land/water boundary
5. Normalize to 16-bit elevation range

Input:  Phase 3 disparity_filtered.tif (uint8), Phase 3 aft_crop.tif
Output: elevation_float.tif (float32), elevation_16bit.tif (uint16), + JPG thumbnails
"""

from PIL import Image
import cv2
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = None

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE3_DIR = os.path.join(BASE_DIR, 'intermediate-processing', 'phase3')
OUTPUT_DIR = os.path.join(BASE_DIR, 'intermediate-processing', 'phase4')


def detect_water_mask(aft_crop_path, target_shape):
    """
    Detect the water body (Lake Nasser) from the aft crop image.
    Water appears as dark, uniform regions in the satellite imagery.
    """
    print("  Detecting water body from aft crop image...")

    aft = np.array(Image.open(aft_crop_path))
    if len(aft.shape) == 3:
        aft = cv2.cvtColor(aft, cv2.COLOR_RGB2GRAY)

    h, w = target_shape

    # Resize aft to match disparity dimensions if needed
    if aft.shape != target_shape:
        aft = cv2.resize(aft, (w, h), interpolation=cv2.INTER_LINEAR)

    # Otsu threshold to separate dark water from bright terrain
    _, binary = cv2.threshold(aft, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    water_mask = binary == 0  # True = dark (water)

    # Dilate to capture shoreline noise
    water_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    water_mask = cv2.dilate(water_mask.astype(np.uint8), water_kernel).astype(bool)

    water_pct = 100 * water_mask.sum() / water_mask.size
    print(f"  Water pixels: {water_mask.sum()} ({water_pct:.1f}%)")

    return water_mask


def inpaint_terrain_gaps(elevation, terrain_mask, water_mask):
    """
    Fill small gaps in TERRAIN ONLY using inpainting.
    Water regions stay at zero.
    """
    print("  Inpainting small terrain gaps...")

    terrain_gap = (~terrain_mask) & (~water_mask)

    if terrain_gap.sum() == 0:
        print("  No terrain gaps to fill")
        return elevation

    t_vals = elevation[terrain_mask]
    if len(t_vals) == 0:
        return elevation

    t_min, t_max = t_vals.min(), t_vals.max()
    t_range = t_max - t_min if t_max > t_min else 1.0

    work = np.zeros_like(elevation, dtype=np.uint8)
    work[terrain_mask] = np.clip(
        ((elevation[terrain_mask] - t_min) / t_range * 254 + 1), 1, 255
    ).astype(np.uint8)

    inpaint_mask = terrain_gap.astype(np.uint8) * 255
    inpainted = cv2.inpaint(work, inpaint_mask, inpaintRadius=10,
                            flags=cv2.INPAINT_TELEA)

    result = elevation.copy()
    gap_values = inpainted[terrain_gap].astype(np.float32)
    result[terrain_gap] = (gap_values - 1) / 254.0 * t_range + t_min

    filled = terrain_gap.sum()
    print(f"  Filled {filled} terrain gap pixels")

    return result


def smooth_terrain(elevation, terrain_mask, water_mask, iterations=3):
    """
    Smooth the terrain surface while:
    - Preserving the sharp land/water cliff edge
    - Keeping water perfectly flat at zero
    """
    print("  Smoothing terrain surface...")

    t_vals = elevation[terrain_mask | ((~water_mask) & (elevation > 0))]
    if len(t_vals) == 0:
        return elevation

    e_min, e_max = t_vals.min(), t_vals.max()
    e_range = e_max - e_min if e_max > e_min else 1.0

    working = np.zeros_like(elevation, dtype=np.uint8)
    nonwater = ~water_mask
    working[nonwater] = np.clip(
        ((elevation[nonwater] - e_min) / e_range * 255), 0, 255
    ).astype(np.uint8)

    for i in range(iterations):
        working = cv2.bilateralFilter(working, d=9, sigmaColor=50, sigmaSpace=50)
        working = cv2.GaussianBlur(working, (5, 5), 0)
        working[water_mask] = 0

    result = working.astype(np.float32) / 255.0 * e_range + e_min
    result[water_mask] = e_min

    print(f"  Smoothed terrain range: {result[nonwater].min():.1f} to {result[nonwater].max():.1f}")
    return result


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
    thumb = thumb.resize((thumb_width, int(thumb_width * h / w)), Image.Resampling.LANCZOS)
    thumb.save(jpg_path, quality=90)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Phase 4: Disparity to Elevation ===\n")

    # Load Phase 3 filtered disparity
    print("Loading filtered disparity map:")
    disparity = np.array(Image.open(os.path.join(PHASE3_DIR, 'disparity_filtered.tif')))
    print(f"  Shape: {disparity.shape}")
    print(f"  Range: {disparity.min()} to {disparity.max()}\n")

    # Step 1: Detect water body from the aft crop (same dimensions as disparity)
    aft_crop_path = os.path.join(PHASE3_DIR, 'aft_crop.tif')
    water_mask = detect_water_mask(aft_crop_path, disparity.shape[:2])

    # Step 2: Build terrain mask
    print("\nBuilding terrain mask:")
    has_data = disparity > 0
    terrain_mask = has_data & (~water_mask)
    # Remove small isolated speckles
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    terrain_uint8 = cv2.morphologyEx(terrain_mask.astype(np.uint8) * 255,
                                      cv2.MORPH_OPEN, open_kernel)
    terrain_mask = terrain_uint8 > 0
    terrain_pct = 100 * terrain_mask.sum() / terrain_mask.size
    print(f"  Valid terrain: {terrain_mask.sum()} / {terrain_mask.size} ({terrain_pct:.1f}%)")

    # Save mask visualization
    mask_vis = np.zeros_like(disparity, dtype=np.uint8)
    mask_vis[terrain_mask] = 200
    mask_vis[water_mask] = 50
    save_with_thumbnail(mask_vis,
                        os.path.join(OUTPUT_DIR, 'validity_mask.tif'),
                        os.path.join(OUTPUT_DIR, 'validity_mask.jpg'))
    print("  Saved: validity_mask.tif/.jpg\n")

    # Step 3: Convert disparity to elevation (float)
    elevation = disparity.astype(np.float32)
    elevation[water_mask] = 0.0

    # Step 4: Inpaint small terrain gaps
    elevation = inpaint_terrain_gaps(elevation, terrain_mask, water_mask)
    print()

    # Step 5: Smooth terrain, keeping water flat
    elevation = smooth_terrain(elevation, terrain_mask, water_mask, iterations=3)
    print()

    # Step 6: Normalize to full range
    print("Normalizing elevation:")
    e_min, e_max = elevation.min(), elevation.max()
    e_range = e_max - e_min if e_max > e_min else 1.0
    elevation_norm = (elevation - e_min) / e_range
    print(f"  Original range: {e_min:.1f} to {e_max:.1f}")
    print(f"  Normalized to: 0.0 to 1.0\n")

    # Save as float32 TIF
    save_with_thumbnail(elevation_norm.astype(np.float32),
                        os.path.join(OUTPUT_DIR, 'elevation_float.tif'),
                        os.path.join(OUTPUT_DIR, 'elevation_float.jpg'))
    print("  Saved: elevation_float.tif/.jpg")

    # Save as 16-bit unsigned integer TIF
    elevation_16 = (elevation_norm * 65535).astype(np.uint16)
    Image.fromarray(elevation_16).save(os.path.join(OUTPUT_DIR, 'elevation_16bit.tif'))
    print("  Saved: elevation_16bit.tif")

    # Save 8-bit preview
    elevation_8 = (elevation_norm * 255).astype(np.uint8)
    save_with_thumbnail(elevation_8,
                        os.path.join(OUTPUT_DIR, 'elevation_preview.tif'),
                        os.path.join(OUTPUT_DIR, 'elevation_preview.jpg'))
    print("  Saved: elevation_preview.tif/.jpg")

    # Save colored elevation
    colored = cv2.applyColorMap(elevation_8, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    save_with_thumbnail(colored_rgb,
                        os.path.join(OUTPUT_DIR, 'elevation_colored.tif'),
                        os.path.join(OUTPUT_DIR, 'elevation_colored.jpg'))
    print("  Saved: elevation_colored.tif/.jpg")

    print(f"\n  Elevation grid size: {elevation_norm.shape[1]}x{elevation_norm.shape[0]}")

    print("\n=== Phase 4 complete ===")


if __name__ == '__main__':
    main()
