# Corona2DEM Project

## Background
Extracting Digital Elevation Models (DEMs) from declassified CORONA KH-4B satellite imagery (1960-1972). The KH-4B carried dual panoramic cameras — one forward (15° ahead), one aft (15° behind) — creating 30° convergent stereo geometry suitable for 3D reconstruction.

## Key Technical Challenges
- Panoramic camera distortions (panoramic, scan positional, image motion compensation)
- No calibration metadata available (no fiducials, lens distortion coefficients, or attitude data)
- Coordinate transforms: scanned image → rotated image → panoramic photo → frame photo → ground
- Need proper overlapping forward/aft stereo pairs from same mission for DEM extraction

## Reference Papers
- Sohn et al. (2004) - "Mathematical modelling of historical reconnaissance CORONA KH-4B Imagery" — three rigorous mathematical methods for modeling KH-4B geometry
- Casana & Cothren (2008) - Practical workflow using sub-images modeled as frame cameras with SPOT/SRTM for GCPs

## USGS M2M API Access
- API Base URL: https://m2m.cr.usgs.gov/api/api/json/stable/
- Auth endpoint: login-token (POST username + token)
- CORONA dataset name: corona2
- Application Token (expires 03/31/2026): DAmMKc40hjhCGLZdlcpIZMPNMrC5kQIRW!XH1GwbBz3FA0MtWCIXGzVGOW!p9SP7
- ERS Username: mikerandrup

## M2M API Usage Notes
- URL must include version: .../json/stable/login-token (not .../json/login-token)
- Auth: POST to login-token with {"username": "...", "token": "..."} — returns session token in "data" field
- Use X-Auth-Token header with session token for subsequent calls
- Use Python (not curl) to avoid shell escaping issues with ! in token
- scene-search spatial filter goes inside sceneFilter wrapper:
  sceneFilter > spatialFilter > filterType: "geojson" > geoJson: {type: "Polygon", coordinates: [...]}
- Metadata filter values for Camera Type: "Forward", "Aft", "Cartographic" (not KH-4B etc.)
- Camera Resolution values: "Stereo High" (KH-4B best), "Stereo Medium" (earlier KH models)
- Missions DS1101+ are generally KH-4B "Stereo High" resolution
- Frame numbers for Forward (DF) and Aft (DA) are offset — the aft camera trails behind, so matching ground coverage has different frame numbers

## CORONA Image Naming Convention
- DS{mission}-{camera_id}{camera_type}{frame}_{subsection}.tif
- DA = Aft camera, DF = Forward camera
- A stereo pair requires matching DA + DF frames from the same mission covering the same ground area

## Linear Project
- Project: Corona2DEM (under "Technical Kickassery" initiative)
- Team: MetroplexWeb
