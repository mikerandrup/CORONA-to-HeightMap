#!/usr/bin/env python3
"""
Phase 5: Export Heightmap
Export the elevation surface in useful formats, preserving the natural shape.

NO square cropping. NO forced resampling to a fixed size.
The output retains the full natural dimensions of the elevation data.

Exports:
- 16-bit RAW file (little-endian, unsigned) for terrain import tools
- 16-bit PNG for general use
- 16-bit TIF for GIS tools
- 8-bit preview JPG and colored visualization

Input: Phase 4 elevation_16bit.tif
Output: heightmap.raw, heightmap.png, heightmap.tif, heightmap_preview.jpg, heightmap_colored.jpg
"""

from PIL import Image
import cv2
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = None

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'intermediate-processing', 'phase4')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Phase 5: Export Heightmap ===\n")

    # Load Phase 4 elevation (16-bit)
    print("Loading elevation data:")
    elevation = np.array(Image.open(os.path.join(INPUT_DIR, 'elevation_16bit.tif')))
    print(f"  Shape: {elevation.shape}, dtype: {elevation.dtype}")
    print(f"  Range: {elevation.min()} to {elevation.max()}")
    print(f"  Natural dimensions: {elevation.shape[1]}x{elevation.shape[0]} (no cropping applied)\n")

    # The elevation data retains its natural shape from the stereo processing.
    # No square cropping or forced resampling.
    heightmap = elevation

    # Save as 16-bit RAW (useful for terrain import tools)
    raw_path = os.path.join(OUTPUT_DIR, 'heightmap.raw')
    heightmap.tofile(raw_path)
    raw_size = os.path.getsize(raw_path)
    print(f"  Saved: heightmap.raw ({raw_size / 1024 / 1024:.1f} MB)")
    print(f"    Dimensions: {heightmap.shape[1]}x{heightmap.shape[0]}")
    print(f"    Bit depth: 16")
    print(f"    Byte order: Little-endian (system native)")

    # Save as 16-bit PNG
    png_path = os.path.join(OUTPUT_DIR, 'heightmap.png')
    Image.fromarray(heightmap).save(png_path)
    print(f"  Saved: heightmap.png")

    # Save as 16-bit TIF
    tif_path = os.path.join(OUTPUT_DIR, 'heightmap.tif')
    Image.fromarray(heightmap).save(tif_path)
    print(f"  Saved: heightmap.tif")

    # Save 8-bit preview JPG (scaled down for reasonable file size)
    preview_8 = (heightmap / 256).astype(np.uint8)
    preview = Image.fromarray(preview_8)
    # Scale down for preview if very large
    w, h = preview.size
    if w > 4000:
        scale = 4000 / w
        preview = preview.resize((4000, int(h * scale)), Image.Resampling.LANCZOS)
    preview.save(os.path.join(OUTPUT_DIR, 'heightmap_preview.jpg'), quality=95)
    print(f"  Saved: heightmap_preview.jpg")

    # Save colored version
    colored = cv2.applyColorMap(preview_8, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored_preview = Image.fromarray(colored_rgb)
    w, h = colored_preview.size
    if w > 4000:
        scale = 4000 / w
        colored_preview = colored_preview.resize((4000, int(h * scale)), Image.Resampling.LANCZOS)
    colored_preview.save(os.path.join(OUTPUT_DIR, 'heightmap_colored.jpg'), quality=95)
    print(f"  Saved: heightmap_colored.jpg")

    print(f"\n  Final heightmap: {heightmap.shape[1]}x{heightmap.shape[0]}, 16-bit")
    print(f"  Natural aspect ratio preserved — no square cropping applied")

    print("\n=== Phase 5 complete ===")
    print("=== Pipeline complete! ===")


if __name__ == '__main__':
    main()
