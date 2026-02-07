#!/usr/bin/env python3
"""
Phase 1: Image Preparation (True Full Resolution)
Load the full CORONA KH-4B stereo sub-images at original resolution.
No cropping. No downsampling. Just trim film scan borders and rotate forward image.

The Aft _b and Forward _c sub-images both contain the Abu Simbel / Lake Nasser area.
The Forward image must be rotated 180° (cameras look in opposite directions).

Input:  Raw CORONA sub-image TIFs (~33-36k x ~11k pixels)
Output: Border-trimmed, rotation-corrected full sub-images
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


def trim_borders(arr, threshold=20):
    """Remove black scan borders from top and bottom of image."""
    row_means = arr.mean(axis=1)
    top = int(np.argmax(row_means > threshold))
    bot = int(arr.shape[0] - np.argmax(row_means[::-1] > threshold) - 1)
    return arr[top:bot, :], top, bot


def save_with_thumbnail(arr, tif_path, jpg_path, thumb_width=2000):
    """Save as TIF and a JPG thumbnail."""
    Image.fromarray(arr).save(tif_path)
    thumb = Image.fromarray(arr)
    w, h = thumb.size
    thumb = thumb.resize((thumb_width, int(thumb_width * h / w)), Image.Resampling.LANCZOS)
    thumb.save(jpg_path, quality=90)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Phase 1: Image Preparation (True Full Resolution) ===\n")

    # Load Aft image at full resolution
    print("Loading Aft image (DA087 _b):")
    print(f"  {AFT_PATH}")
    img = Image.open(AFT_PATH)
    print(f"  Original size: {img.size[0]}x{img.size[1]}")
    aft = np.array(img)
    img.close()
    aft, aft_top, aft_bot = trim_borders(aft)
    print(f"  After trim: {aft.shape} (removed top {aft_top} rows, bottom from {aft_bot})")

    save_with_thumbnail(aft,
                        os.path.join(OUTPUT_DIR, 'aft_full.tif'),
                        os.path.join(OUTPUT_DIR, 'aft_full.jpg'))
    print(f"  Saved: aft_full.tif/.jpg\n")
    del aft

    # Load Forward image at full resolution, rotate 180°
    print("Loading Forward image (DF081 _c):")
    print(f"  {FWD_PATH}")
    img = Image.open(FWD_PATH)
    print(f"  Original size: {img.size[0]}x{img.size[1]}")
    fwd = np.array(img)
    img.close()
    fwd, fwd_top, fwd_bot = trim_borders(fwd)
    print(f"  After trim: {fwd.shape}")
    fwd = np.rot90(fwd, 2)
    print(f"  Rotated 180°: {fwd.shape}")

    save_with_thumbnail(fwd,
                        os.path.join(OUTPUT_DIR, 'fwd_full.tif'),
                        os.path.join(OUTPUT_DIR, 'fwd_full.jpg'))
    print(f"  Saved: fwd_full.tif/.jpg")

    print("\n=== Phase 1 complete ===")


if __name__ == '__main__':
    main()
