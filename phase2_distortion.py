#!/usr/bin/env python3
"""
Phase 2: Distortion Correction (Full Resolution)
Apply simplified panoramic-to-frame photo coordinate transforms from Sohn et al. (2004).

The CORONA KH-4B panoramic camera introduces geometric distortions not present in
frame cameras. The dominant effect is panoramic distortion caused by the cylindrical
film surface and scanning lens. We correct this to produce frame-equivalent imagery
suitable for standard stereo matching.

Simplified approach (demo quality):
- Apply panoramic distortion correction: xf = f * tan(α), yf = yp * sec(α)
  where α is the scan angle from the panoramic photo x-coordinate
- Skip scan positional distortion and IMC (secondary effects)

KH-4B parameters:
- Focal length f = 609.602 mm
- Film scan resolution d = 7 μm per pixel at full resolution
- The panoramic scan sweeps ±35° from nadir (70° total field of view)

This version operates at full resolution — pixel_scale = 1.0 for full-res crops.
"""

from PIL import Image
import cv2
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = None

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'intermediate-processing', 'phase1')
OUTPUT_DIR = os.path.join(BASE_DIR, 'intermediate-processing', 'phase2')

# KH-4B camera parameters
FOCAL_LENGTH_MM = 609.602  # mm
SCAN_RESOLUTION_UM = 7.0   # μm per pixel at full resolution


def correct_panoramic_distortion(arr, pixel_scale=1.0):
    """
    Apply panoramic distortion correction to convert from panoramic photo
    coordinates to frame photo coordinates.

    Processes in vertical strips to avoid OpenCV's SHRT_MAX (32767) limit
    on image dimensions in cv2.remap.

    Args:
        arr: 2D grayscale image array (panoramic photo)
        pixel_scale: ratio of actual pixel size to full-res pixel size (1.0 = full res)
    Returns:
        corrected: distortion-corrected image array
    """
    h, w = arr.shape

    # Effective pixel size at our resolution (mm per pixel)
    d_eff = SCAN_RESOLUTION_UM * 1e-3 * pixel_scale  # convert μm to mm, scale

    # Angular resolution per pixel (radians)
    delta_alpha = d_eff / FOCAL_LENGTH_MM

    # Center of the panoramic image
    xc = w / 2.0
    yc = h / 2.0

    # Determine output image size
    alpha_max = (w / 2.0) * delta_alpha
    xf_max = FOCAL_LENGTH_MM * np.tan(alpha_max) / d_eff
    out_w = int(2 * xf_max) + 1
    out_h = h

    print(f"  Max scan angle: {np.degrees(alpha_max):.1f}°")
    print(f"  Output size: {out_w}x{out_h} (from {w}x{h})")

    # Precompute the 1D mapping arrays (these define the full correction)
    out_xc = out_w / 2.0
    xf_grid = np.arange(out_w, dtype=np.float32) - out_xc
    yf_grid = np.arange(out_h, dtype=np.float32) - yc

    # Inverse: α = arctan(xf_px * d_eff / f)
    alpha = np.arctan(xf_grid * d_eff / FOCAL_LENGTH_MM)
    src_x = alpha / delta_alpha + xc  # source x for each output column
    cos_alpha = np.cos(alpha)

    # Process in vertical strips to stay under SHRT_MAX (32767) for cv2.remap.
    # Both source and destination must be < 32767 in each dimension.
    STRIP_WIDTH = 16000  # output columns per strip
    corrected = np.zeros((out_h, out_w), dtype=arr.dtype)

    n_strips = (out_w + STRIP_WIDTH - 1) // STRIP_WIDTH
    print(f"  Processing in {n_strips} vertical strip(s) (SHRT_MAX workaround)...")

    for strip_idx in range(n_strips):
        c0 = strip_idx * STRIP_WIDTH
        c1 = min(c0 + STRIP_WIDTH, out_w)
        strip_w = c1 - c0

        # Determine which source columns this output strip needs
        strip_src_x = src_x[c0:c1]
        src_col_min = max(0, int(np.floor(strip_src_x.min())) - 2)
        src_col_max = min(w, int(np.ceil(strip_src_x.max())) + 3)

        # Extract the source sub-image for this strip
        arr_strip = arr[:, src_col_min:src_col_max]

        # Adjust map_x to be relative to the extracted source sub-image
        map_x = np.tile(strip_src_x - src_col_min, (out_h, 1)).astype(np.float32)
        map_y = np.zeros((out_h, strip_w), dtype=np.float32)

        for col in range(strip_w):
            map_y[:, col] = (yf_grid * cos_alpha[c0 + col] + yc).astype(np.float32)

        # Remap this strip using the cropped source
        strip = cv2.remap(arr_strip, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        corrected[:, c0:c1] = strip

        print(f"    Strip {strip_idx+1}/{n_strips}: output cols [{c0}:{c1}], "
              f"source cols [{src_col_min}:{src_col_max}]")

        del map_x, map_y, strip, arr_strip

    return corrected


def save_with_thumbnail(arr, tif_path, jpg_path, thumb_width=1000):
    """Save as TIF and a JPG thumbnail."""
    Image.fromarray(arr).save(tif_path)
    thumb = Image.fromarray(arr)
    w, h = thumb.size
    thumb = thumb.resize((thumb_width, int(thumb_width * h / w)), Image.Resampling.LANCZOS)
    thumb.save(jpg_path, quality=90)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Phase 2: Distortion Correction (Full Resolution) ===\n")

    # At full resolution, pixel_scale = 1.0 (each pixel = 7μm on film)
    pixel_scale = 1.0

    for name, label in [('aft_full.tif', 'Aft'), ('fwd_full.tif', 'Fwd')]:
        print(f"Processing {label} image:")
        path = os.path.join(INPUT_DIR, name)
        arr = np.array(Image.open(path))
        print(f"  Input: {arr.shape}")
        print(f"  Pixel scale: {pixel_scale:.1f}x (full resolution, 7μm/pixel)")

        corrected = correct_panoramic_distortion(arr, pixel_scale)

        # Cap width back to SHRT_MAX after distortion correction (which expands width)
        h_c, w_c = corrected.shape
        if w_c > 32767:
            excess = w_c - 32767
            left = excess // 2
            corrected = corrected[:, left:left + 32767]
            print(f"  Width capped: {w_c} -> 32767 (removed {left}px from each edge)")

        out_name = name.replace('.tif', '_corrected')
        tif_path = os.path.join(OUTPUT_DIR, out_name + '.tif')
        jpg_path = os.path.join(OUTPUT_DIR, out_name + '.jpg')
        save_with_thumbnail(corrected, tif_path, jpg_path)
        print(f"  Saved: {out_name}.tif/.jpg\n")

    # Save before/after comparison thumbnail
    aft_before = np.array(Image.open(os.path.join(INPUT_DIR, 'aft_full.tif')))
    aft_after = np.array(Image.open(os.path.join(OUTPUT_DIR, 'aft_full_corrected.tif')))

    target_h = min(aft_before.shape[0], aft_after.shape[0])
    before_resized = cv2.resize(aft_before, (int(aft_before.shape[1] * target_h / aft_before.shape[0]), target_h))
    after_resized = cv2.resize(aft_after, (int(aft_after.shape[1] * target_h / aft_after.shape[0]), target_h))

    gap = 10
    comp_w = before_resized.shape[1] + after_resized.shape[1] + gap
    comparison = np.zeros((target_h, comp_w), dtype=np.uint8)
    comparison[:, :before_resized.shape[1]] = before_resized
    comparison[:, before_resized.shape[1] + gap:] = after_resized

    comp_thumb = Image.fromarray(comparison)
    cw, ch = comp_thumb.size
    comp_thumb = comp_thumb.resize((2000, int(2000 * ch / cw)), Image.Resampling.LANCZOS)
    comp_thumb.save(os.path.join(OUTPUT_DIR, 'before_after_comparison.jpg'), quality=90)
    print("Saved: before_after_comparison.jpg")

    print("\n=== Phase 2 complete ===")


if __name__ == '__main__':
    main()
