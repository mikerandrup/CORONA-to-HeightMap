#!/usr/bin/env python3
"""
Phase 5: Export Heightmap
Export the elevation surface as highest-quality JPG.

Input: Phase 4 elevation_16bit.tif
Output: heightmap.jpg
"""

from PIL import Image
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'intermediate-processing', 'phase4')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Phase 5: Export Heightmap ===\n")

    print("Loading elevation data:")
    elevation = np.array(
        Image.open(os.path.join(INPUT_DIR, 'elevation_16bit.tif'))
    )
    print(f"  Shape: {elevation.shape}, dtype: {elevation.dtype}")
    print(f"  Range: {elevation.min()} to {elevation.max()}")

    elevation_8bit = (elevation / 256).astype(np.uint8)
    print(f"  Converted to 8-bit: {elevation_8bit.min()} to {elevation_8bit.max()}")

    jpg_path = os.path.join(OUTPUT_DIR, 'heightmap.jpg')
    Image.fromarray(elevation_8bit).save(jpg_path, quality=100, subsampling=0)
    print(f"  Saved: heightmap.jpg (quality=100)")
    print(f"  Dimensions: {elevation.shape[1]}x{elevation.shape[0]}")

    print("\n=== Phase 5 complete ===")
    print("=== Pipeline complete! ===")


if __name__ == '__main__':
    main()
