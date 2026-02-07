# Corona2DEM

Extracting Digital Elevation Models from declassified CORONA KH-4B satellite imagery.

## Background

The CORONA program (1960–1972) was the United States' first operational space reconnaissance system. The KH-4B missions, which operated from 1967–1972, carried dual panoramic cameras that captured stereo imagery suitable for 3D terrain reconstruction. These images were declassified in 1995 and are now available through the USGS.

This project implements a pipeline to extract relative elevation data from KH-4B stereo pairs, producing heightmaps that reveal terrain that may have changed dramatically—or been submerged entirely—in the decades since acquisition.

## The KH-4B Camera System

The KH-4B carried two panoramic cameras on a common mount:

- **Forward camera**: pointed 15° ahead of vertical
- **Aft camera**: pointed 15° behind vertical
- **Convergent angle**: 30° between views
- **Focal length**: 609.6mm
- **Flying height**: ~145km
- **Ground resolution**: ~1.8m at nadir
- **Scan angle**: 70° (217km swath width)

The 30° convergent geometry creates stereo parallax—the horizontal displacement between corresponding points in the forward and aft images encodes elevation information.

## Theoretical Foundation

This pipeline draws on techniques from two key papers:

### Sohn et al. (2004)
*"Mathematical modelling of historical reconnaissance CORONA KH-4B Imagery"*
The Photogrammetric Record 19(105): 51–66

Sohn et al. describe three rigorous mathematical methods for correcting the geometric distortions inherent in KH-4B panoramic imagery:

1. **Panoramic distortion** — caused by the cylindrical film surface and scanning lens action
2. **Scan positional distortion** — caused by satellite motion during the ~0.5 second scan
3. **Image Motion Compensation (IMC) distortion** — caused by lens/film translation

The key coordinate transform from panoramic photo coordinates to equivalent frame photo coordinates:

```
xf = f · tan(α)
yf = yp · sec(α)
```

where `α` is the scan angle and `f` is the focal length.

### Casana & Cothren (2008)
*"Stereo analysis, DEM extraction and orthorectification of CORONA satellite imagery: archaeological applications from the Near East"*
Antiquity 82: 1–18

Casana & Cothren demonstrate practical DEM extraction by treating small sub-images as frame cameras—a simplification that introduces ~5 pixel error but enables use of standard photogrammetric tools. They achieved 10m DEMs suitable for mapping archaeological sites.

**Critical insight**: The along-track displacement between forward and aft views IS the elevation signal. Any geometric correction that removes this horizontal offset destroys the stereo parallax.

## Pipeline Overview

Our pipeline processes a KH-4B stereo pair through five phases:

| Phase | Purpose | Key Output |
|-------|---------|------------|
| 1 | Image Preparation | Trimmed, oriented full-resolution images |
| 2 | Distortion Correction | Panoramic distortion removed |
| 3 | Stereo Matching | Disparity map from SGBM |
| 4 | Elevation Extraction | Relative elevation surface |
| 5 | Export | Final heightmap |

---

## Phase 1: Image Preparation

Prepares the raw scanned CORONA frames for processing:

- Trims film edges and sprocket holes
- Caps width to 32767 pixels (OpenCV SHRT_MAX limit)
- Rotates forward image 180° (forward and aft cameras face opposite directions)
- Outputs full-resolution grayscale images

### Aft Image (DA087)
![Aft full](intermediate-processing/phase1/aft_full.jpg)

### Forward Image (DF081)
![Forward full](intermediate-processing/phase1/fwd_full.jpg)

---

## Phase 2: Distortion Correction

Corrects the panoramic distortion inherent in the KH-4B scanning camera system.

The panoramic camera exposes film on a cylindrical platen as the lens rotates through a 70° arc. This creates a "bow-tie" distortion where the image scale varies across the scan direction. Following Sohn et al., we apply the inverse transform:

```
x_corrected = f · tan(x_panoramic / f)
```

This converts the cylindrical projection to an equivalent flat (frame) projection while preserving the stereo geometry.

### Before/After Comparison
![Before after comparison](intermediate-processing/phase2/before_after_comparison.jpg)

### Aft Corrected
![Aft corrected](intermediate-processing/phase2/aft_full_corrected.jpg)

### Forward Corrected
![Forward corrected](intermediate-processing/phase2/fwd_full_corrected.jpg)

---

## Phase 3: Stereo Matching

This is the critical phase where elevation information is extracted from stereo parallax.

### Why No Homography

A previous version of this pipeline used `cv2.findHomography()` to align the forward and aft images before stereo matching. This was **catastrophically wrong**—the 8-DOF projective transform absorbed the along-track disparity that encodes elevation, producing flat results.

**The corrected approach:**

1. Use SIFT feature matching to find the overlap region and measure offsets
2. Measure the **along-track offset (dx)** — this is the stereo baseline, ~1887 pixels
3. Measure the **cross-track offset (dy)** — this is scanline misalignment, ~1272 pixels
4. Apply **only the vertical shift** to align scanlines
5. **Preserve the horizontal offset** — this IS the stereo parallax
6. Run Semi-Global Block Matching (SGBM) on the aligned pair

### Stereo Overlay
The overlay shows the aligned stereo pair. Slight color fringing indicates parallax variation from terrain relief—exactly what we want to see.

![Stereo overlay](intermediate-processing/phase3/stereo_overlay.jpg)

### Cropped Aft Image
![Aft crop](intermediate-processing/phase3/aft_crop.jpg)

### Cropped Forward Image
![Forward crop](intermediate-processing/phase3/fwd_crop.jpg)

### Raw Disparity
![Disparity raw](intermediate-processing/phase3/disparity_raw.jpg)

### Filtered Disparity
![Disparity filtered](intermediate-processing/phase3/disparity_filtered.jpg)

### Colored Disparity
![Disparity colored](intermediate-processing/phase3/disparity_colored.jpg)

---

## Phase 4: Elevation Extraction

Converts disparity values to a relative elevation surface.

### Processing Steps

1. **Water detection** — identifies water bodies (Lake Nasser) from the aft crop image using intensity thresholding
2. **Terrain masking** — separates valid terrain from water and invalid disparity regions
3. **Gap inpainting** — fills small holes in the terrain surface
4. **Smoothing** — applies bilateral filtering to reduce noise while preserving edges
5. **Normalization** — scales elevation to 0–1 range

### Validity Mask
Blue indicates water (Lake Nasser), gray indicates valid terrain.

![Validity mask](intermediate-processing/phase4/validity_mask.jpg)

### Elevation Preview
![Elevation preview](intermediate-processing/phase4/elevation_preview.jpg)

### Colored Elevation
![Elevation colored](intermediate-processing/phase4/elevation_colored.jpg)

---

## Phase 5: Export

Exports the final heightmap as a highest-quality JPG for visualization and use in other applications.

### Final Heightmap
![Heightmap](output/heightmap.jpg)

---

## Test Data

The pipeline was developed using imagery from mission **DS1105-2** covering the **Abu Simbel** region of southern Egypt:

- **Aft frame**: DS1105-2235DA087_87_b.tif
- **Forward frame**: DS1105-2235DF081_81_c.tif
- **Acquisition date**: November 4, 1968

This region includes:
- The Nile River valley
- Lake Nasser (formed by the Aswan High Dam, completed 1970)
- Desert terrain with significant topographic relief
- The Abu Simbel temples (relocated 1964–1968 due to dam construction)

## Limitations

This pipeline produces **relative** elevation data, not calibrated absolute elevations:

- No ground control points (GCPs) are used
- No exterior orientation modeling
- Disparity is treated as proportional to elevation without rigorous calibration

For archaeological survey and terrain visualization, relative elevation is often sufficient. Absolute elevation would require GCPs from GPS surveys or orthorectified reference imagery, as described in Casana & Cothren (2008).

## References

1. Sohn, H.G., Kim, G.H., & Yom, J.H. (2004). Mathematical modelling of historical reconnaissance CORONA KH-4B Imagery. *The Photogrammetric Record* 19(105): 51–66.

2. Casana, J. & Cothren, J. (2008). Stereo analysis, DEM extraction and orthorectification of CORONA satellite imagery: archaeological applications from the Near East. *Antiquity* 82: 1–18.

3. National Reconnaissance Office (1967). *The KH-4B Camera System*.

## USGS Data Access

CORONA imagery is available from the USGS EarthExplorer:
- Dataset: `corona2`
- API: https://m2m.cr.usgs.gov/api/api/json/stable/
