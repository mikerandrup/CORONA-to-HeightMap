#!/usr/bin/env python3
"""
Phase 3: Stereo Matching (Full Resolution)
Generate a disparity map from the distortion-corrected Forward/Aft stereo pair.

Approach:
1. Use SIFT feature matching to compute an AFFINE transform (not homography)
   that aligns the two images. Affine has fewer degrees of freedom than
   homography, so it cannot absorb the parallax signal as easily.
   This preserves the elevation-encoding disparity while removing rotation,
   scale, and shear differences between the views.
2. Apply OpenCV's StereoSGBM (Semi-Global Block Matching) for dense disparity
3. Filter and smooth the disparity map

At full resolution (~10000px wide), the parallax differences should be
multiple pixels, giving SGBM enough signal to resolve terrain elevation.

Input: Phase 2 corrected TIFs (full resolution)
Output: Disparity map as TIF + JPG thumbnail
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


def align_stereo_pair(aft, fwd):
    """
    Use feature matching to warp the Forward image into alignment with the Aft image.
    Uses homography (8 DOF) for alignment — this provides the best geometric fit
    between the two views, minimizing vertical misalignment so that SGBM can
    search along horizontal scanlines effectively.

    The parallax signal may be partially absorbed by the homography on flat terrain,
    but at full resolution the remaining disparity differences should still be
    measurable, especially near terrain features with real elevation change.
    """
    print("  Computing stereo alignment (homography)...")

    # Equalize for better matching
    aft_eq = cv2.equalizeHist(aft)
    fwd_eq = cv2.equalizeHist(fwd)

    # SIFT matching — use more features at full resolution
    sift = cv2.SIFT_create(nfeatures=15000)
    kp1, des1 = sift.detectAndCompute(aft_eq, None)
    kp2, des2 = sift.detectAndCompute(fwd_eq, None)
    print(f"  Keypoints: aft={len(kp1)}, fwd={len(kp2)}")

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good = [m for m, n in raw_matches if m.distance < 0.7 * n.distance]
    print(f"  Good matches: {len(good)}")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Homography: warp fwd onto aft coordinate system
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)
    inliers = mask.ravel().astype(bool).sum()
    print(f"  RANSAC inliers: {inliers}")

    # Warp forward image to align with aft
    h, w = aft.shape
    fwd_warped = cv2.warpPerspective(fwd, H, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)

    # Compute residual displacement stats on inlier matches
    inlier_idx = mask.ravel().astype(bool)
    inlier_src = np.float32([kp1[good[i].queryIdx].pt for i in range(len(good)) if inlier_idx[i]])
    inlier_dst_orig = np.float32([kp2[good[i].trainIdx].pt for i in range(len(good)) if inlier_idx[i]])

    inlier_dst_warped = cv2.perspectiveTransform(
        inlier_dst_orig.reshape(-1, 1, 2), H
    ).reshape(-1, 2)

    residual = inlier_src - inlier_dst_warped
    print(f"  Residual after warp: dx={residual[:,0].mean():.2f}±{residual[:,0].std():.2f}, "
          f"dy={residual[:,1].mean():.2f}±{residual[:,1].std():.2f}")
    print(f"  Residual range: dx=[{residual[:,0].min():.1f}, {residual[:,0].max():.1f}], "
          f"dy=[{residual[:,1].min():.1f}, {residual[:,1].max():.1f}]")

    return fwd_warped


def compute_disparity(aft, fwd_warped):
    """
    Compute dense disparity map using Semi-Global Block Matching.
    The aft image is the 'left' image and the warped forward image is the 'right'.

    Processes in horizontal strips to fit within RAM. SGBM operates row-by-row
    so horizontal strips are natural. Strips overlap vertically so the SGBM
    cost aggregation paths have context; only the non-overlapping center rows
    are kept from each strip.
    """
    print("  Computing disparity map (SGBM) in horizontal strips...")

    h, w = aft.shape

    # SGBM parameters
    min_disp = -64
    num_disp = 192
    block_size = 15

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 1 * block_size ** 2,
        P2=32 * 1 * block_size ** 2,
        disp12MaxDiff=-1,
        uniquenessRatio=1,
        speckleWindowSize=400,
        speckleRange=4,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    print(f"  Disparity search range: [{min_disp}, {min_disp + num_disp}]")
    print(f"  Block size: {block_size}")
    print(f"  Image size: {w}x{h}")

    # Strip parameters — sized to keep memory reasonable on 36GB machine
    # Each strip: w * strip_h * num_disp * ~4 bytes for SGBM internals
    STRIP_HEIGHT = 512    # core rows per strip (keeps SGBM under ~13GB)
    OVERLAP = 64          # overlap rows on each side for SGBM path context

    disparity_full = np.zeros((h, w), dtype=np.float32)
    n_strips = (h + STRIP_HEIGHT - 1) // STRIP_HEIGHT
    print(f"  Processing {n_strips} horizontal strips ({STRIP_HEIGHT}px + {OVERLAP}px overlap)...")

    for i in range(n_strips):
        core_r0 = i * STRIP_HEIGHT
        core_r1 = min(core_r0 + STRIP_HEIGHT, h)

        # Expand with overlap for SGBM context
        strip_r0 = max(0, core_r0 - OVERLAP)
        strip_r1 = min(h, core_r1 + OVERLAP)

        aft_strip = aft[strip_r0:strip_r1, :]
        fwd_strip = fwd_warped[strip_r0:strip_r1, :]

        disp_raw = stereo.compute(aft_strip, fwd_strip)
        disp_strip = disp_raw.astype(np.float32) / 16.0

        # Extract only the core (non-overlap) rows
        local_core_r0 = core_r0 - strip_r0
        local_core_r1 = local_core_r0 + (core_r1 - core_r0)
        disparity_full[core_r0:core_r1, :] = disp_strip[local_core_r0:local_core_r1, :]

        del aft_strip, fwd_strip, disp_raw, disp_strip

        print(f"    Strip {i+1}/{n_strips}: rows [{core_r0}:{core_r1}]")

    # Mask invalid disparities
    valid_mask = disparity_full > (min_disp - 1)
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
    """
    print("  Filtering disparity map...")

    valid_vals = disparity[valid_mask]
    if len(valid_vals) == 0:
        return disparity

    d_min = valid_vals.min()
    d_max = valid_vals.max()
    d_range = d_max - d_min if d_max > d_min else 1.0

    # Normalize to 0-255
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

    # Light Gaussian blur
    smoothed = cv2.GaussianBlur(smoothed, (5, 5), 0)

    print(f"  Filtered disparity range: {smoothed.min()} to {smoothed.max()}")

    return smoothed


def save_with_thumbnail(arr, tif_path, jpg_path, thumb_width=1000):
    """Save as TIF and a JPG thumbnail."""
    Image.fromarray(arr).save(tif_path)
    thumb = Image.fromarray(arr)
    w, h = thumb.size
    thumb = thumb.resize((thumb_width, int(thumb_width * h / w)), Image.Resampling.LANCZOS)
    thumb.save(jpg_path, quality=90)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Phase 3: Stereo Matching (Full Resolution) ===\n")

    # Load Phase 2 outputs and downsample to half resolution for memory
    print("Loading corrected images:")
    aft_full = np.array(Image.open(os.path.join(INPUT_DIR, 'aft_full_corrected.tif')))
    fwd_full = np.array(Image.open(os.path.join(INPUT_DIR, 'fwd_full_corrected.tif')))
    print(f"  Aft full: {aft_full.shape}")
    print(f"  Fwd full: {fwd_full.shape}")

    # Downsample to half resolution — full res OOMs during SIFT/SGBM on 36GB
    aft = cv2.resize(aft_full, (aft_full.shape[1] // 2, aft_full.shape[0] // 2), interpolation=cv2.INTER_AREA)
    fwd = cv2.resize(fwd_full, (fwd_full.shape[1] // 2, fwd_full.shape[0] // 2), interpolation=cv2.INTER_AREA)
    del aft_full, fwd_full
    print(f"  Aft half: {aft.shape}")
    print(f"  Fwd half: {fwd.shape}\n")

    # Step 1: Align stereo pair
    fwd_warped = align_stereo_pair(aft, fwd)

    # Save alignment result
    save_with_thumbnail(fwd_warped,
                        os.path.join(OUTPUT_DIR, 'fwd_warped.tif'),
                        os.path.join(OUTPUT_DIR, 'fwd_warped.jpg'))
    print("  Saved: fwd_warped.tif/.jpg\n")

    # Save overlay to verify alignment
    overlay = cv2.addWeighted(aft, 0.5, fwd_warped, 0.5, 0)
    save_with_thumbnail(overlay,
                        os.path.join(OUTPUT_DIR, 'stereo_overlay.tif'),
                        os.path.join(OUTPUT_DIR, 'stereo_overlay.jpg'))
    print("  Saved: stereo_overlay.tif/.jpg\n")

    # Step 2: Compute disparity
    disparity, valid_mask = compute_disparity(aft, fwd_warped)

    # Save raw disparity visualization
    raw_vis = np.zeros_like(disparity, dtype=np.uint8)
    if valid_mask.sum() > 0:
        v = disparity[valid_mask]
        raw_vis[valid_mask] = np.clip(
            (disparity[valid_mask] - v.min()) / (v.max() - v.min()) * 255, 0, 255
        ).astype(np.uint8)
    save_with_thumbnail(raw_vis,
                        os.path.join(OUTPUT_DIR, 'disparity_raw.tif'),
                        os.path.join(OUTPUT_DIR, 'disparity_raw.jpg'))
    print("  Saved: disparity_raw.tif/.jpg\n")

    # Step 3: Filter disparity
    filtered = filter_disparity(disparity, valid_mask)
    save_with_thumbnail(filtered,
                        os.path.join(OUTPUT_DIR, 'disparity_filtered.tif'),
                        os.path.join(OUTPUT_DIR, 'disparity_filtered.jpg'))
    print("  Saved: disparity_filtered.tif/.jpg\n")

    # Save colorized version
    colored = cv2.applyColorMap(filtered, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    save_with_thumbnail(colored_rgb,
                        os.path.join(OUTPUT_DIR, 'disparity_colored.tif'),
                        os.path.join(OUTPUT_DIR, 'disparity_colored.jpg'))
    print("  Saved: disparity_colored.tif/.jpg")

    print("\n=== Phase 3 complete ===")


if __name__ == '__main__':
    main()
