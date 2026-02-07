#!/usr/bin/env python3
"""
Phase 3: Stereo Matching (Corrected — No Homography)

The KH-4B forward and aft cameras are rigidly mounted on the same satellite,
same focal length, same altitude, scanning simultaneously. After Phase 2
panoramic distortion correction, the ONLY geometric difference between the
two views is the 30° convergence angle producing along-track parallax.

There is NO rotation, scale, or shear to correct — both cameras share the
same rigid platform and identical optics. A homography or even affine warp
would absorb the parallax signal we need.

Approach:
1. Use SIFT matching to find the overlap region and measure offsets
2. Crop both images to overlapping area
3. Apply ONLY a vertical (cross-track) integer shift to align scanlines
4. Run StereoSGBM for dense disparity
5. Filter and save

Input: Phase 2 corrected TIFs (full resolution)
Output: Disparity map as float32 TIF + visualization JPGs
"""

from PIL import Image
import cv2
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = None

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'intermediate-processing', 'phase2')
OUTPUT_DIR = os.path.join(BASE_DIR, 'intermediate-processing', 'phase3')


def find_overlap_and_offsets(aft, fwd):
    """
    Use SIFT feature matching to determine:
    - The mean horizontal (along-track) offset between the images
    - The mean vertical (cross-track) offset between the images
    - The overlapping region bounds

    Returns:
        dx_mean: mean horizontal offset (along-track parallax baseline)
        dy_mean: mean vertical offset (cross-track misalignment to correct)
        overlap_cols_aft: (col_start, col_end) in aft image coordinates
        overlap_cols_fwd: (col_start, col_end) in fwd image coordinates
    """
    print("  Finding overlap region via SIFT matching...")

    # Downsample for faster matching (just for finding the offset)
    scale = 0.25
    aft_small = cv2.resize(aft, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    fwd_small = cv2.resize(fwd, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    aft_eq = cv2.equalizeHist(aft_small)
    fwd_eq = cv2.equalizeHist(fwd_small)

    sift = cv2.SIFT_create(nfeatures=10000)
    kp1, des1 = sift.detectAndCompute(aft_eq, None)
    kp2, des2 = sift.detectAndCompute(fwd_eq, None)
    print(f"  Keypoints: aft={len(kp1)}, fwd={len(kp2)}")

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        raise RuntimeError("Not enough keypoints for matching")

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good = [m for m, n in raw_matches if m.distance < 0.7 * n.distance]
    print(f"  Good matches: {len(good)}")

    if len(good) < 20:
        raise RuntimeError(f"Only {len(good)} good matches — need at least 20")

    # Get matched point coordinates (scaled back to full resolution)
    pts_aft = np.float32([kp1[m.queryIdx].pt for m in good]) / scale
    pts_fwd = np.float32([kp2[m.trainIdx].pt for m in good]) / scale

    # Compute offsets: aft_point = fwd_point + offset
    dx = pts_aft[:, 0] - pts_fwd[:, 0]  # horizontal (along-track)
    dy = pts_aft[:, 1] - pts_fwd[:, 1]  # vertical (cross-track)

    # Use median for robustness against outliers
    dx_median = np.median(dx)
    dy_median = np.median(dy)

    # Filter outliers (beyond 3 MAD from median)
    dx_mad = np.median(np.abs(dx - dx_median))
    dy_mad = np.median(np.abs(dy - dy_median))
    inlier = (np.abs(dx - dx_median) < 5 * max(dx_mad, 1)) & \
             (np.abs(dy - dy_median) < 5 * max(dy_mad, 1))

    dx_mean = np.mean(dx[inlier])
    dy_mean = np.mean(dy[inlier])
    dx_std = np.std(dx[inlier])
    dy_std = np.std(dy[inlier])

    print(f"  Inlier matches: {inlier.sum()}")
    print(f"  Along-track offset (dx): {dx_mean:.1f} ± {dx_std:.1f} pixels")
    print(f"  Cross-track offset (dy): {dy_mean:.1f} ± {dy_std:.1f} pixels")
    print(f"  (dx is the stereo baseline — this is PRESERVED, not corrected)")

    return dx_mean, dy_mean, pts_aft[inlier], pts_fwd[inlier]


def crop_to_overlap(aft, fwd, dx_mean, dy_mean):
    """
    Crop both images to their overlapping region.
    Apply ONLY a vertical (cross-track) integer pixel shift to align scanlines.
    The horizontal offset is NOT corrected — it IS the stereo parallax.
    """
    print("\n  Cropping to overlap region...")

    h_aft, w_aft = aft.shape
    h_fwd, w_fwd = fwd.shape

    dy_int = int(round(dy_mean))
    dx_int = int(round(dx_mean))

    print(f"  Aft size: {w_aft}x{h_aft}")
    print(f"  Fwd size: {w_fwd}x{h_fwd}")
    print(f"  Applying vertical shift: {dy_int}px (cross-track alignment)")
    print(f"  NOT applying horizontal shift: {dx_int}px (this is stereo parallax)")

    # Horizontal overlap: if dx_mean > 0, aft is shifted right relative to fwd
    # So the overlap in aft starts at max(0, dx_int) and in fwd starts at max(0, -dx_int)
    if dx_int >= 0:
        aft_col_start = dx_int
        fwd_col_start = 0
    else:
        aft_col_start = 0
        fwd_col_start = -dx_int

    overlap_w = min(w_aft - aft_col_start, w_fwd - fwd_col_start)

    # Vertical overlap accounting for dy shift
    if dy_int >= 0:
        aft_row_start = dy_int
        fwd_row_start = 0
    else:
        aft_row_start = 0
        fwd_row_start = -dy_int

    overlap_h = min(h_aft - aft_row_start, h_fwd - fwd_row_start)

    # Crop both to overlap region
    aft_crop = aft[aft_row_start:aft_row_start + overlap_h,
                   aft_col_start:aft_col_start + overlap_w]
    fwd_crop = fwd[fwd_row_start:fwd_row_start + overlap_h,
                   fwd_col_start:fwd_col_start + overlap_w]

    print(f"  Overlap size: {overlap_w}x{overlap_h}")
    print(f"  Aft crop: rows [{aft_row_start}:{aft_row_start + overlap_h}], "
          f"cols [{aft_col_start}:{aft_col_start + overlap_w}]")
    print(f"  Fwd crop: rows [{fwd_row_start}:{fwd_row_start + overlap_h}], "
          f"cols [{fwd_col_start}:{fwd_col_start + overlap_w}]")

    return aft_crop, fwd_crop


def compute_disparity(aft_crop, fwd_crop):
    """
    Compute dense disparity map using Semi-Global Block Matching.
    The aft image is 'left', the fwd image is 'right'.

    The images have been cropped to overlap and vertically aligned, but the
    horizontal stereo parallax is preserved — that's what SGBM will measure.

    Processes in horizontal strips to manage memory.
    """
    print("\n  Computing disparity map (SGBM) in horizontal strips...")

    h, w = aft_crop.shape

    # SGBM parameters — search range must cover the expected parallax
    # The feature matching dx_mean tells us the approximate disparity center
    # We search a wide range around it
    min_disp = 0
    num_disp = 256  # must be divisible by 16
    block_size = 11

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 1 * block_size ** 2,
        P2=32 * 1 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=200,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    print(f"  Disparity search range: [{min_disp}, {min_disp + num_disp}]")
    print(f"  Block size: {block_size}")
    print(f"  Image size: {w}x{h}")

    # Strip parameters for memory management
    STRIP_HEIGHT = 512
    OVERLAP = 64

    disparity_full = np.zeros((h, w), dtype=np.float32)
    n_strips = (h + STRIP_HEIGHT - 1) // STRIP_HEIGHT
    print(f"  Processing {n_strips} horizontal strips ({STRIP_HEIGHT}px + {OVERLAP}px overlap)...")

    for i in range(n_strips):
        core_r0 = i * STRIP_HEIGHT
        core_r1 = min(core_r0 + STRIP_HEIGHT, h)

        strip_r0 = max(0, core_r0 - OVERLAP)
        strip_r1 = min(h, core_r1 + OVERLAP)

        aft_strip = aft_crop[strip_r0:strip_r1, :]
        fwd_strip = fwd_crop[strip_r0:strip_r1, :]

        disp_raw = stereo.compute(aft_strip, fwd_strip)
        disp_strip = disp_raw.astype(np.float32) / 16.0

        local_core_r0 = core_r0 - strip_r0
        local_core_r1 = local_core_r0 + (core_r1 - core_r0)
        disparity_full[core_r0:core_r1, :] = disp_strip[local_core_r0:local_core_r1, :]

        del aft_strip, fwd_strip, disp_raw, disp_strip

        if (i + 1) % 10 == 0 or i == n_strips - 1:
            print(f"    Strip {i+1}/{n_strips}: rows [{core_r0}:{core_r1}]")

    # Stats on valid disparities
    valid_mask = disparity_full > min_disp
    print(f"  Valid pixels: {valid_mask.sum()} / {valid_mask.size} "
          f"({100*valid_mask.sum()/valid_mask.size:.1f}%)")

    if valid_mask.sum() > 0:
        valid_vals = disparity_full[valid_mask]
        print(f"  Disparity range: {valid_vals.min():.1f} to {valid_vals.max():.1f}")
        print(f"  Disparity mean: {valid_vals.mean():.1f}, std: {valid_vals.std():.1f}")

    return disparity_full, valid_mask


def filter_disparity(disparity, valid_mask):
    """
    Clean up the disparity map:
    - Fill small gaps via morphological closing
    - Apply bilateral filter to smooth while preserving edges
    Returns both the filtered uint8 visualization and the float32 disparity.
    """
    print("\n  Filtering disparity map...")

    valid_vals = disparity[valid_mask]
    if len(valid_vals) == 0:
        print("  WARNING: No valid disparity values!")
        return disparity, np.zeros_like(disparity, dtype=np.uint8)

    d_min = valid_vals.min()
    d_max = valid_vals.max()
    d_range = d_max - d_min if d_max > d_min else 1.0

    # Normalize to 0-255 for visualization and filtering
    normalized = np.zeros_like(disparity)
    normalized[valid_mask] = ((disparity[valid_mask] - d_min) / d_range * 255).astype(np.float32)

    # Fill invalid pixels with morphological closing
    filled = normalized.copy()
    invalid = ~valid_mask
    if invalid.sum() > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        filled_uint8 = filled.astype(np.uint8)
        closed = cv2.morphologyEx(filled_uint8, cv2.MORPH_CLOSE, kernel)
        filled[invalid] = closed[invalid].astype(np.float32)

    # Bilateral filter for edge-preserving smoothing
    filled_uint8 = np.clip(filled, 0, 255).astype(np.uint8)
    smoothed = cv2.bilateralFilter(filled_uint8, d=9, sigmaColor=75, sigmaSpace=75)

    print(f"  Filtered disparity range: {smoothed.min()} to {smoothed.max()}")

    return disparity, smoothed


def save_with_thumbnail(arr, tif_path, jpg_path, thumb_width=1000):
    """Save as TIF and a JPG thumbnail."""
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        # Save TIF with float data
        Image.fromarray(arr).save(tif_path)
        # Convert to uint8 for JPG
        a_min, a_max = arr.min(), arr.max()
        a_range = a_max - a_min if a_max > a_min else 1.0
        arr_8 = ((arr - a_min) / a_range * 255).astype(np.uint8)
    else:
        Image.fromarray(arr).save(tif_path)
        arr_8 = arr

    thumb = Image.fromarray(arr_8)
    w, h = thumb.size
    thumb = thumb.resize((thumb_width, int(thumb_width * h / w)), Image.Resampling.LANCZOS)
    thumb.save(jpg_path, quality=90)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Phase 3: Stereo Matching (No Homography — Preserving Parallax) ===\n")

    # Load Phase 2 outputs
    print("Loading corrected images:")
    aft = np.array(Image.open(os.path.join(INPUT_DIR, 'aft_full_corrected.tif')))
    fwd = np.array(Image.open(os.path.join(INPUT_DIR, 'fwd_full_corrected.tif')))
    print(f"  Aft: {aft.shape}")
    print(f"  Fwd: {fwd.shape}\n")

    # Step 1: Find overlap and measure offsets
    dx_mean, dy_mean, pts_aft, pts_fwd = find_overlap_and_offsets(aft, fwd)

    # Step 2: Crop to overlap with vertical-only alignment
    aft_crop, fwd_crop = crop_to_overlap(aft, fwd, dx_mean, dy_mean)
    del aft, fwd  # free memory

    # Save cropped images for inspection
    save_with_thumbnail(aft_crop,
                        os.path.join(OUTPUT_DIR, 'aft_crop.tif'),
                        os.path.join(OUTPUT_DIR, 'aft_crop.jpg'))
    save_with_thumbnail(fwd_crop,
                        os.path.join(OUTPUT_DIR, 'fwd_crop.tif'),
                        os.path.join(OUTPUT_DIR, 'fwd_crop.jpg'))
    print("  Saved: aft_crop.tif/.jpg, fwd_crop.tif/.jpg")

    # Save overlay for visual verification
    overlay = cv2.addWeighted(aft_crop, 0.5, fwd_crop, 0.5, 0)
    save_with_thumbnail(overlay,
                        os.path.join(OUTPUT_DIR, 'stereo_overlay.tif'),
                        os.path.join(OUTPUT_DIR, 'stereo_overlay.jpg'))
    print("  Saved: stereo_overlay.tif/.jpg\n")
    del overlay

    # Step 3: Compute disparity
    disparity_raw, valid_mask = compute_disparity(aft_crop, fwd_crop)

    # Save raw disparity (float32 — preserves full precision for Phase 4)
    save_with_thumbnail(disparity_raw,
                        os.path.join(OUTPUT_DIR, 'disparity_raw.tif'),
                        os.path.join(OUTPUT_DIR, 'disparity_raw.jpg'))
    print("  Saved: disparity_raw.tif/.jpg\n")

    # Step 4: Filter disparity
    disparity_float, filtered_uint8 = filter_disparity(disparity_raw, valid_mask)
    save_with_thumbnail(filtered_uint8,
                        os.path.join(OUTPUT_DIR, 'disparity_filtered.tif'),
                        os.path.join(OUTPUT_DIR, 'disparity_filtered.jpg'))
    print("  Saved: disparity_filtered.tif/.jpg\n")

    # Save colorized version
    colored = cv2.applyColorMap(filtered_uint8, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    save_with_thumbnail(colored_rgb,
                        os.path.join(OUTPUT_DIR, 'disparity_colored.tif'),
                        os.path.join(OUTPUT_DIR, 'disparity_colored.jpg'))
    print("  Saved: disparity_colored.tif/.jpg")

    print("\n=== Phase 3 complete ===")


if __name__ == '__main__':
    main()
