#!/usr/bin/env python3
"""
Phase 1: Image Preparation
Extract matching sub-regions from CORONA KH-4B stereo pair for Abu Simbel area.

Key findings from exploration:
- Sub-images a,b,c,d tile top-to-bottom along the flight direction
- The Aft _b and Forward _c sub-images both contain Lake Nasser / Abu Simbel
- The Forward image must be rotated 180° to align with the Aft image
  (cameras look in opposite directions)
- After rotation, SIFT matching confirms 415 inliers with tight displacement
- The stereo parallax is ~120px horizontal at low-res (2000px wide)
"""

from PIL import Image
import cv2
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = None

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'corona_images', 'DS1105_abu_simbel')
OUTPUT_DIR = os.path.join(BASE_DIR, 'intermediate-processing', 'phase1')

AFT_PATH = os.path.join(IMG_DIR, 'DS1105-2235DA087_87_b.tif')
FWD_PATH = os.path.join(IMG_DIR, 'DS1105-2235DF081_81_c.tif')

# Working resolution for processing (keeps memory manageable)
WORK_WIDTH = 4000

# Target crop size for stereo processing
CROP_SIZE = 4000


def trim_borders(arr, threshold=20):
    """Remove black scan borders from top and bottom of image."""
    row_means = arr.mean(axis=1)
    top = int(np.argmax(row_means > threshold))
    bot = int(arr.shape[0] - np.argmax(row_means[::-1] > threshold) - 1)
    return arr[top:bot, :], top, bot


def load_and_prepare(path, rotate180=False):
    """Load a CORONA sub-image, trim borders, optionally rotate 180°."""
    print(f"  Loading {os.path.basename(path)}...")
    img = Image.open(path)
    full_w, full_h = img.size
    print(f"  Full size: {full_w}x{full_h}")

    # Resize to working resolution
    scale = WORK_WIDTH / full_w
    work_h = int(full_h * scale)
    img_resized = img.resize((WORK_WIDTH, work_h), Image.Resampling.LANCZOS)
    arr = np.array(img_resized)
    img.close()

    # Trim black borders
    arr, top, bot = trim_borders(arr)
    print(f"  After trim: {arr.shape} (removed rows {top} and below {bot})")

    if rotate180:
        arr = np.rot90(arr, 2)  # 180° rotation
        print(f"  Rotated 180°")

    return arr, scale


def find_overlap_and_crop(aft, fwd):
    """Use SIFT + RANSAC to find the overlapping region and extract matching crops."""

    # Equalize histograms for matching
    aft_eq = cv2.equalizeHist(aft)
    fwd_eq = cv2.equalizeHist(fwd)

    # SIFT feature matching
    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(aft_eq, None)
    kp2, des2 = sift.detectAndCompute(fwd_eq, None)
    print(f"  SIFT keypoints: aft={len(kp1)}, fwd={len(kp2)}")

    # FLANN matching with ratio test
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good = [m for m, n in raw_matches if m.distance < 0.7 * n.distance]
    print(f"  Good matches (ratio test): {len(good)}")

    # RANSAC homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = mask.ravel().astype(bool)
    print(f"  RANSAC inliers: {inliers.sum()} / {len(good)}")

    # Get inlier correspondences
    inlier_src = np.float32([kp1[good[i].queryIdx].pt for i in range(len(good)) if inliers[i]])
    inlier_dst = np.float32([kp2[good[i].trainIdx].pt for i in range(len(good)) if inliers[i]])

    # Find the overlapping bounding box in both images
    # Use the median of matched points to find center of overlap
    aft_cx = np.median(inlier_src[:, 0])
    aft_cy = np.median(inlier_src[:, 1])
    fwd_cx = np.median(inlier_dst[:, 0])
    fwd_cy = np.median(inlier_dst[:, 1])
    print(f"  Overlap center - Aft: ({aft_cx:.0f}, {aft_cy:.0f}), Fwd: ({fwd_cx:.0f}, {fwd_cy:.0f})")

    # Extract crops centered on the overlap center
    half = CROP_SIZE // 2

    def safe_crop(arr, cx, cy):
        cx, cy = int(cx), int(cy)
        r0 = max(0, cy - half)
        r1 = min(arr.shape[0], cy + half)
        c0 = max(0, cx - half)
        c1 = min(arr.shape[1], cx + half)
        return arr[r0:r1, c0:c1], c0, r0

    aft_crop, aft_c0, aft_r0 = safe_crop(aft, aft_cx, aft_cy)
    fwd_crop, fwd_c0, fwd_r0 = safe_crop(fwd, fwd_cx, fwd_cy)
    print(f"  Aft crop: {aft_crop.shape} from ({aft_c0}, {aft_r0})")
    print(f"  Fwd crop: {fwd_crop.shape} from ({fwd_c0}, {fwd_r0})")

    return aft_crop, fwd_crop


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Phase 1: Image Preparation ===\n")

    print("Loading Aft image (DA087 _b):")
    aft, aft_scale = load_and_prepare(AFT_PATH, rotate180=False)
    print(f"  Working image: {aft.shape}\n")

    print("Loading Forward image (DF081 _c):")
    fwd, fwd_scale = load_and_prepare(FWD_PATH, rotate180=True)
    print(f"  Working image: {fwd.shape}\n")

    print("Finding overlap region and extracting crops:")
    aft_crop, fwd_crop = find_overlap_and_crop(aft, fwd)

    # Save outputs
    aft_path = os.path.join(OUTPUT_DIR, 'aft_crop.tif')
    fwd_path = os.path.join(OUTPUT_DIR, 'fwd_crop.tif')

    Image.fromarray(aft_crop).save(aft_path)
    Image.fromarray(fwd_crop).save(fwd_path)
    print(f"\n  Saved: {aft_path}")
    print(f"  Saved: {fwd_path}")

    # Save preview JPGs
    for arr, name in [(aft_crop, 'aft_crop_preview.jpg'), (fwd_crop, 'fwd_crop_preview.jpg')]:
        preview = Image.fromarray(arr)
        w, h = preview.size
        preview = preview.resize((1000, int(1000 * h / w)), Image.Resampling.LANCZOS)
        preview.save(os.path.join(OUTPUT_DIR, name), quality=90)
        print(f"  Saved: {name}")

    # Save a side-by-side comparison
    h = max(aft_crop.shape[0], fwd_crop.shape[0])
    w = aft_crop.shape[1] + fwd_crop.shape[1] + 20
    comparison = np.zeros((h, w), dtype=np.uint8)
    comparison[:aft_crop.shape[0], :aft_crop.shape[1]] = aft_crop
    comparison[:fwd_crop.shape[0], aft_crop.shape[1]+20:] = fwd_crop
    comp_preview = Image.fromarray(comparison)
    comp_preview = comp_preview.resize((2000, int(2000 * h / w)), Image.Resampling.LANCZOS)
    comp_preview.save(os.path.join(OUTPUT_DIR, 'stereo_comparison.jpg'), quality=90)
    print(f"  Saved: stereo_comparison.jpg")

    print("\n=== Phase 1 complete ===")


if __name__ == '__main__':
    main()
