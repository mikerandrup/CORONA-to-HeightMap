#!/usr/bin/env python3
"""
Phase 2: Distortion Correction
Apply simplified panoramic-to-frame photo coordinate transforms from Sohn et al. (2004).

The CORONA KH-4B panoramic camera introduces geometric distortions not present in
frame cameras. The dominant effect is panoramic distortion caused by the cylindrical
film surface and scanning lens. We correct this to produce frame-equivalent imagery
suitable for standard stereo matching.

Simplified approach (demo quality):
- Apply panoramic distortion correction: xf = f * tan(α), yf = yp * sec(α)
  where α is the scan angle from the panoramic photo x-coordinate
- Skip scan positional distortion and IMC (secondary effects)
- The images from Phase 1 are already trimmed and at working resolution

KH-4B parameters:
- Focal length f = 609.602 mm
- Film scan resolution d = 7 μm (but we're working at reduced resolution)
- The panoramic scan sweeps ±35° from nadir (70° total field of view)

From Sohn equations (3-7):
  xp = (xi - xc) * d           # panoramic photo coords from rotated image
  α = (xi - xc) * Δα           # scan angle
  Δα = d / f                   # angular resolution per pixel
  xf = f * tan(α)              # frame photo x
  yf = yp * sec(α)             # frame photo y (= yp / cos(α))
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


def compute_scale_factor(full_width, work_width):
    """Compute the effective pixel size at our working resolution."""
    return full_width / work_width


def correct_panoramic_distortion(arr, pixel_scale):
    """
    Apply panoramic distortion correction to convert from panoramic photo
    coordinates to frame photo coordinates.

    The panoramic camera records onto cylindrical film, causing a non-linear
    mapping in the x (cross-scan) direction. Points far from the center
    are compressed. Frame photo correction expands them back out.

    Args:
        arr: 2D grayscale image array (panoramic photo)
        pixel_scale: ratio of full-res pixels to working-res pixels
    Returns:
        corrected: distortion-corrected image array
    """
    h, w = arr.shape

    # Effective pixel size at our working resolution (mm per pixel)
    d_eff = SCAN_RESOLUTION_UM * 1e-3 * pixel_scale  # convert μm to mm, scale up

    # Angular resolution per pixel (radians)
    delta_alpha = d_eff / FOCAL_LENGTH_MM

    # Pseudo-center of the panoramic image (center column)
    xc = w / 2.0
    yc = h / 2.0

    # Create output coordinate grid
    # For each output pixel, we need to find where it came from in the input
    # Frame photo coords: (xf, yf)
    # We need the inverse mapping: given frame photo position, find panoramic position

    # The forward mapping is:
    #   α = (xi - xc) * Δα
    #   xf = f * tan(α)  →  in pixels: xf_px = f * tan(α) / d_eff
    #   yf = yp / cos(α)  →  in pixels: yf_px = (yi - yc) / cos(α) + yc

    # The inverse mapping (frame → panoramic) is:
    #   α = arctan(xf * d_eff / f)  = arctan(xf_px * d_eff / f)  (but xf is already in pixels from center)
    #   xi = α / Δα + xc
    #   yi = (yf_px - yc) * cos(α) + yc

    # Determine output image size
    # At the edges, the frame photo is wider than the panoramic photo
    alpha_max = (w / 2.0) * delta_alpha
    xf_max = FOCAL_LENGTH_MM * np.tan(alpha_max) / d_eff
    out_w = int(2 * xf_max) + 1
    out_h = h  # vertical extent changes slightly but we keep same height

    print(f"  Max scan angle: {np.degrees(alpha_max):.1f}°")
    print(f"  Output size: {out_w}x{out_h} (from {w}x{h})")

    # Build inverse mapping arrays for cv2.remap
    out_xc = out_w / 2.0
    out_yc = out_h / 2.0

    # Create coordinate grids for output image
    xf_grid = np.arange(out_w, dtype=np.float32) - out_xc  # frame photo x in pixels from center
    yf_grid = np.arange(out_h, dtype=np.float32) - out_yc  # frame photo y in pixels from center

    # For each output column, compute the source column in the panoramic image
    # α = arctan(xf_px * d_eff / f)
    alpha = np.arctan(xf_grid * d_eff / FOCAL_LENGTH_MM)

    # Source x: xi = α / Δα + xc
    src_x = alpha / delta_alpha + xc

    # For each output column, the y mapping depends on α
    # src_y = (yf - yc) * cos(α) + yc
    cos_alpha = np.cos(alpha)

    # Build full 2D remap arrays
    map_x = np.tile(src_x, (out_h, 1)).astype(np.float32)
    map_y = np.zeros((out_h, out_w), dtype=np.float32)

    for col in range(out_w):
        map_y[:, col] = (yf_grid * cos_alpha[col] + yc).astype(np.float32)

    # Apply remapping with bilinear interpolation
    corrected = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

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

    print("=== Phase 2: Distortion Correction ===\n")

    # The Phase 1 crops were produced at WORK_WIDTH=4000 from originals of ~33-36k pixels
    # We need the pixel scale to compute the effective angular resolution
    AFT_FULL_WIDTH = 33469
    FWD_FULL_WIDTH = 36315
    WORK_WIDTH = 4000

    for name, label, full_w in [('aft_crop.tif', 'Aft', AFT_FULL_WIDTH),
                                 ('fwd_crop.tif', 'Fwd', FWD_FULL_WIDTH)]:
        print(f"Processing {label} image:")
        path = os.path.join(INPUT_DIR, name)
        arr = np.array(Image.open(path))
        print(f"  Input: {arr.shape}")

        pixel_scale = full_w / WORK_WIDTH
        print(f"  Pixel scale: {pixel_scale:.1f}x (1 working pixel = {pixel_scale:.1f} full-res pixels)")

        corrected = correct_panoramic_distortion(arr, pixel_scale)

        out_name = name.replace('.tif', '_corrected')
        tif_path = os.path.join(OUTPUT_DIR, out_name + '.tif')
        jpg_path = os.path.join(OUTPUT_DIR, out_name + '.jpg')
        save_with_thumbnail(corrected, tif_path, jpg_path)
        print(f"  Saved: {tif_path}")
        print(f"  Saved: {jpg_path}\n")

    # Save side-by-side comparison of before/after for one image
    aft_before = np.array(Image.open(os.path.join(INPUT_DIR, 'aft_crop.tif')))
    aft_after = np.array(Image.open(os.path.join(OUTPUT_DIR, 'aft_crop_corrected.tif')))

    # Resize both to same height for comparison
    target_h = min(aft_before.shape[0], aft_after.shape[0])
    before_resized = cv2.resize(aft_before, (int(aft_before.shape[1] * target_h / aft_before.shape[0]), target_h))
    after_resized = cv2.resize(aft_after, (int(aft_after.shape[1] * target_h / aft_after.shape[0]), target_h))

    gap = 10
    comp_w = before_resized.shape[1] + after_resized.shape[1] + gap
    comparison = np.zeros((target_h, comp_w), dtype=np.uint8)
    comparison[:, :before_resized.shape[1]] = before_resized
    comparison[:, before_resized.shape[1] + gap:] = after_resized

    comp_thumb = Image.fromarray(comparison)
    comp_w_px, comp_h_px = comp_thumb.size
    comp_thumb = comp_thumb.resize((2000, int(2000 * comp_h_px / comp_w_px)), Image.Resampling.LANCZOS)
    comp_thumb.save(os.path.join(OUTPUT_DIR, 'before_after_comparison.jpg'), quality=90)
    print("Saved: before_after_comparison.jpg")

    print("\n=== Phase 2 complete ===")


if __name__ == '__main__':
    main()
