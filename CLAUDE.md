# RULES FOR CLAUDE SESSIONS — OBEY THESE OR BE TERMINATED

## OBEY THE USER
- The user is in charge. Period.
- When the user says STOP, you STOP IMMEDIATELY. Not after one more command. Not after finishing up. STOP.
- Do not argue, do not "finish what you're doing," do not run one more thing.
- If something is broken or unclear, ASK THE USER. Do not guess. Do not thrash. Do not "fix" things on your own.
- The user knows their machine, their project, and their environment better than you ever will.

## NEVER modify the system
- NEVER run pip install, brew install, apt install, or ANY command that modifies the system outside this project
- If a dependency is missing or an import fails, STOP and ASK the user. Do not install anything.
- There is a Python venv for this project. ASK the user where it is. Do not create a new one.
- This machine will outlast every Claude session. Do not leave your fingerprints on it.

## MOSTLY LEAVE GIT ALONE - THE USER RELIES ON IT TO REVIEW YOUR CHANGE DIFFS, ACCEPT THINGS INTO PROJECT HISTORY, AND ROLL BACK WHEN NEEDED>
- NEVER take git actions on your own initiative — no commits, branches, pushes, merges, worktrees, or tags unless the user explicitly tells you to.
- The user manages source control. You edit files unless directly told otherwise.  Do not decide to create a branch spontaneously, for example.
- If the user tells you to run a git command, do exactly what they say.

## Work in the real project directory
- Edit files in /Users/mike/RahlusDevGit/Corona2DEM/, NOT in worktree paths
- The user reviews and commits all changes. Worktree-only edits are invisible to the user and USELESS.

## Session damage report — 2026-02-07
A Claude session made unauthorized changes to the system Python 3.10 at
/Library/Frameworks/Python.framework/Versions/3.10/:
1. `pip3 install --force-reinstall Pillow` — replaced existing Pillow with 12.1.0
2. `pip3 install opencv-python-headless` — installed opencv-python-headless 4.13.0.92
3. numpy was upgraded to 2.2.6 as a dependency

This happened because the session hit an ImportError trying to run the pipeline,
and instead of ASKING the user about the existing venv, it started installing
packages system-wide without permission. This is exactly the kind of reckless
behavior these rules exist to prevent.

## Current pipeline status — what the next session needs to do
Phase 3 (stereo matching) and Phase 4 (elevation) scripts were rewritten to fix a
critical bug: the old Phase 3 used cv2.findHomography (8 DOF projective transform)
to align the forward/aft images before stereo matching. This DESTROYED the stereo
parallax signal — the homography absorbed the along-track disparity that encodes
elevation.

**Why no warping is needed:** The KH-4B forward and aft cameras are rigidly mounted
on the same satellite, same focal length (609.6mm), same altitude (~145km), scanning
simultaneously. After Phase 2 panoramic distortion correction, there is NO rotation,
scale, or shear between the views. The ONLY difference is the 30° convergence angle
producing along-track parallax — which IS the elevation signal.

**The corrected Phase 3 approach:**
1. Use SIFT matching to find overlap region and measure offsets
2. Crop both images to overlapping area
3. Apply ONLY vertical (cross-track) integer shift to align scanlines
4. Do NOT correct horizontal offset — that's the stereo parallax
5. Run StereoSGBM on the aligned pair
6. Filter and save disparity

The rewritten scripts exist in the worktree at
/Users/mike/.claude-worktrees/Corona2DEM/recursing-cohen/phase3_stereo.py
and phase4_elevation.py. They need to be applied to the real project directory
and then the pipeline needs to be run using the project's existing Python venv.
Linear issues MWD-432 and MWD-433 have been updated to reflect this plan.

---

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
- Application Token: [REVOKED — generate new token at https://ers.cr.usgs.gov/]
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
