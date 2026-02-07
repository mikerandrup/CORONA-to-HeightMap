#!/usr/bin/env python3
"""
Phase 4: Disparity to Elevation
Convert the filtered disparity map into a clean relative-elevation surface.

Approach:
1. Load the filtered disparity from Phase 3
2. Detect water body (Lake Nasser) from the aft image — force to flat zero elevation
   Note: disparity noise over water is NOT terrain — it's film exposure variation
   from minor cloud cover and shallow water reflections. Must be clamped flat.
3. Mask out no-overlap borders (bottom portion where forward image didn't cover)
4. Inpaint only small terrain gaps (speckle holes in valid land areas)
5. Force water regions back to flat zero AFTER inpainting (prevent bleed-in)
6. Smooth terrain while preserving the sharp land/water boundary
7. Normalize to 16-bit elevation range

Input:  Phase 3 disparity_filtered.tif (uint8)
Output: elevation_float.tif (float32), elevation_16bit.tif (uint16), + JPG thumbnails
"""

from PIL import Image
import cv2
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = None

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'intermediate-processing', 'phase3')
OUTPUT_DIR = os.path.join(BASE_DIR, 'intermediate-processing', 'phase4')


def detect_water_mask(aft_image_path, target_shape):
    """
    Detect the water body (Lake Nasser) from the aft image.
    Water appears as dark, uniform regions in the satellite imagery.
    Any disparity signal over water is film artifact, not terrain.
    """
    print("  Detecting water body from aft image...")

    aft = np.array(Image.open(aft_image_path))
    h, w = target_shape

    # Resize aft to match disparity dimensions if needed
    if aft.shape != target_shape:
        aft = cv2.resize(aft, (w, h), interpolation=cv2.INTER_LINEAR)

    # Otsu threshold to separate dark water from bright terrain
    _, binary = cv2.threshold(aft, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    water_mask = binary == 0  # True = dark (water)

    # Dilate generously to capture shoreline noise and film artifacts
    # along the water boundary — these are NOT real terrain
    water_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    water_mask = cv2.dilate(water_mask.astype(np.uint8), water_kernel).astype(bool)

    water_pct = 100 * water_mask.sum() / water_mask.size
    print(f"  Water pixels: {water_mask.sum()} ({water_pct:.1f}%)")

    return water_mask


def detect_overlap_bounds(disparity):
    """
    Find the region where both stereo images have valid data.
    The bottom portion of the image has no forward-image coverage.
    """
    print("  Detecting stereo overlap bounds...")

    h, w = disparity.shape

    # Find rows/cols where enough pixels have non-zero disparity
    row_valid_frac = (disparity > 0).mean(axis=1)
    col_valid_frac = (disparity > 0).mean(axis=0)

    valid_rows = row_valid_frac > 0.2
    valid_cols = col_valid_frac > 0.2

    if valid_rows.any():
        first_row = int(np.where(valid_rows)[0][0])
        last_row = int(np.where(valid_rows)[0][-1])
    else:
        first_row, last_row = 0, h - 1

    if valid_cols.any():
        first_col = int(np.where(valid_cols)[0][0])
        last_col = int(np.where(valid_cols)[0][-1])
    else:
        first_col, last_col = 0, w - 1

    print(f"  Overlap region: rows [{first_row}:{last_row}], cols [{first_col}:{last_col}]")
    print(f"  Overlap size: {last_col - first_col + 1}x{last_row - first_row + 1}")

    return first_row, last_row + 1, first_col, last_col + 1


def build_terrain_mask(disparity, water_mask, crop_bounds):
    """
    Build a mask of pixels that represent valid terrain (not water, not no-data).
    """
    r0, r1, c0, c1 = crop_bounds
    disp_crop = disparity[r0:r1, c0:c1]
    water_crop = water_mask[r0:r1, c0:c1]

    # Valid terrain = has disparity data AND is not water
    has_data = disp_crop > 0
    terrain = has_data & (~water_crop)

    # Remove small isolated speckles
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    terrain_uint8 = cv2.morphologyEx(terrain.astype(np.uint8) * 255,
                                      cv2.MORPH_OPEN, open_kernel)
    terrain = terrain_uint8 > 0

    terrain_pct = 100 * terrain.sum() / terrain.size
    print(f"  Valid terrain in crop: {terrain.sum()} / {terrain.size} ({terrain_pct:.1f}%)")

    return terrain, water_crop


def inpaint_terrain_gaps(elevation, terrain_mask, water_mask):
    """
    Fill small gaps in TERRAIN ONLY using inpainting.
    Water regions are NOT inpainted — they stay at zero.
    """
    print("  Inpainting small terrain gaps...")

    # Only inpaint holes that are within the land area (not water)
    # A "terrain gap" is a pixel that is NOT terrain AND NOT water
    terrain_gap = (~terrain_mask) & (~water_mask)

    if terrain_gap.sum() == 0:
        print("  No terrain gaps to fill")
        return elevation

    # Normalize terrain values to uint8 for inpainting
    t_vals = elevation[terrain_mask]
    if len(t_vals) == 0:
        return elevation

    t_min, t_max = t_vals.min(), t_vals.max()
    t_range = t_max - t_min if t_max > t_min else 1.0

    work = np.zeros_like(elevation, dtype=np.uint8)
    work[terrain_mask] = np.clip(
        ((elevation[terrain_mask] - t_min) / t_range * 254 + 1), 1, 255
    ).astype(np.uint8)

    # Inpaint only terrain gaps (not water)
    inpaint_mask = terrain_gap.astype(np.uint8) * 255
    inpainted = cv2.inpaint(work, inpaint_mask, inpaintRadius=10,
                            flags=cv2.INPAINT_TELEA)

    # Convert back to float, but only update the gap pixels
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

    # Normalize to 0-255 for filtering
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
        # Bilateral filter — preserves edges (cliffs) while smoothing terrain
        working = cv2.bilateralFilter(working, d=9, sigmaColor=50, sigmaSpace=50)
        # Light Gaussian
        working = cv2.GaussianBlur(working, (5, 5), 0)
        # Force water back to zero after each pass
        working[water_mask] = 0

    # Convert back to float
    result = working.astype(np.float32) / 255.0 * e_range + e_min
    # Final enforcement: water = minimum elevation (flat lake surface)
    result[water_mask] = e_min

    print(f"  Smoothed terrain range: {result[nonwater].min():.1f} to {result[nonwater].max():.1f}")
    return result


def save_with_thumbnail(arr, tif_path, jpg_path, thumb_width=1000):
    """Save as TIF and a JPG thumbnail."""
    Image.fromarray(arr).save(tif_path)
    # For JPG thumbnail, convert float/16-bit to 8-bit
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
    disparity = np.array(Image.open(os.path.join(INPUT_DIR, 'disparity_filtered.tif')))
    print(f"  Shape: {disparity.shape}")
    print(f"  Range: {disparity.min()} to {disparity.max()}\n")

    # Step 1: Detect water body
    aft_path = os.path.join(BASE_DIR, 'intermediate-processing', 'phase2', 'aft_crop_corrected.tif')
    water_mask = detect_water_mask(aft_path, disparity.shape)

    # Step 2: Find stereo overlap bounds
    crop_bounds = detect_overlap_bounds(disparity)
    r0, r1, c0, c1 = crop_bounds

    # Step 3: Build terrain mask
    print("\nBuilding terrain mask:")
    terrain_mask, water_crop = build_terrain_mask(disparity, water_mask, crop_bounds)

    # Save full-size mask visualization
    mask_vis = np.zeros_like(disparity, dtype=np.uint8)
    mask_vis_crop = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
    mask_vis_crop[terrain_mask] = 200     # valid terrain = bright
    mask_vis_crop[water_crop] = 50         # water = dark gray
    mask_vis[r0:r1, c0:c1] = mask_vis_crop
    save_with_thumbnail(mask_vis,
                        os.path.join(OUTPUT_DIR, 'validity_mask.tif'),
                        os.path.join(OUTPUT_DIR, 'validity_mask.jpg'))
    print("  Saved: validity_mask.tif/.jpg\n")

    # Step 4: Crop and process
    print(f"Cropping to overlap region: {c1-c0}x{r1-r0}")
    disp_crop = disparity[r0:r1, c0:c1].astype(np.float32)

    # Zero out water disparity — it's film artifact, not terrain
    disp_crop[water_crop] = 0.0
    print(f"  Zeroed water pixels in disparity\n")

    # Step 5: Inpaint only small terrain gaps
    elevation = inpaint_terrain_gaps(disp_crop, terrain_mask, water_crop)
    print()

    # Step 6: Smooth terrain, keeping water flat
    elevation = smooth_terrain(elevation, terrain_mask, water_crop, iterations=3)
    print()

    # Step 7: Normalize to full range
    print("Normalizing elevation:")
    e_min, e_max = elevation.min(), elevation.max()
    e_range = e_max - e_min if e_max > e_min else 1.0
    elevation_norm = (elevation - e_min) / e_range
    print(f"  Original range: {e_min:.1f} to {e_max:.1f}")
    print(f"  Normalized to: 0.0 to 1.0")
    print(f"  Water level in normalized space: {(0 - e_min) / e_range:.3f}\n")

    # Save as float32 TIF (full precision)
    save_with_thumbnail(elevation_norm.astype(np.float32),
                        os.path.join(OUTPUT_DIR, 'elevation_float.tif'),
                        os.path.join(OUTPUT_DIR, 'elevation_float.jpg'))
    print("  Saved: elevation_float.tif/.jpg")

    # Save as 16-bit unsigned integer TIF (65535 levels — enough for Unity)
    elevation_16 = (elevation_norm * 65535).astype(np.uint16)
    Image.fromarray(elevation_16).save(os.path.join(OUTPUT_DIR, 'elevation_16bit.tif'))
    print("  Saved: elevation_16bit.tif")

    # Save 8-bit preview
    elevation_8 = (elevation_norm * 255).astype(np.uint8)
    save_with_thumbnail(elevation_8,
                        os.path.join(OUTPUT_DIR, 'elevation_preview.tif'),
                        os.path.join(OUTPUT_DIR, 'elevation_preview.jpg'))
    print("  Saved: elevation_preview.tif/.jpg")

    # Save colored elevation for visual inspection
    colored = cv2.applyColorMap(elevation_8, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    save_with_thumbnail(colored_rgb,
                        os.path.join(OUTPUT_DIR, 'elevation_colored.tif'),
                        os.path.join(OUTPUT_DIR, 'elevation_colored.jpg'))
    print("  Saved: elevation_colored.tif/.jpg")

    print(f"\n  Elevation grid size: {elevation_norm.shape[1]}x{elevation_norm.shape[0]}")
    print(f"  Ready for Phase 5 resampling to 2048x2048")

    print("\n=== Phase 4 complete ===")


if __name__ == '__main__':
    main()
