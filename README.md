# UAV Aerial Mapping Pipeline — Production Build

## What this is

A working photogrammetry pipeline that reconstructs orthomosaics, DSMs, and
dense point clouds from overlapping aerial photos. It replaces a previous
from-scratch OpenCV implementation that was structurally broken on real UAV data.

## Why the previous pipeline was abandoned

The previous build tried to reimplement Structure-from-Motion, Multi-View
Stereo, and aerial mosaicking in pure OpenCV+NumPy. Three of those choices
were structurally wrong for real aerial data:

1. **Incremental SfM without proper bundle adjustment.** The previous code
   ran a single LM optimization on the seed pair only (80 points, 2 cameras).
   The other 76 cameras and 13,561 points were never refined. This caused
   the entire camera path to collapse onto a tiny line in the visible output.

2. **Two-view StereoSGBM on consecutive pairs as MVS.** Real multi-view
   stereo uses PatchMatch-based methods with view selection (PMVS, OpenMVS,
   COLMAP MVS). Two-view SGBM on near-nadir aerial pairs with small baselines
   produces noise at coordinate ranges of ±200,000 — exactly what your
   Image 4 showed.

3. **`cv2.Stitcher` for the orthomosaic.** Stitcher is a panorama tool that
   estimates a global rotational camera model. It is the wrong algorithm
   class for nadir aerial mosaicking, which requires per-pixel orthorectification
   using the DSM. Even if it had run without OOM, the result would not have
   been a valid orthomosaic. Image 6 is what fundamentally-wrong-algorithm
   output looks like.

Patching these three independent issues would mean reimplementing COLMAP +
OpenMVS + ODM. That is multi-month research engineering, not a graduation
deliverable.

## What was rebuilt

The reconstruction core is now [OpenDroneMap (ODM)](https://opendronemap.org/),
the open-source photogrammetry standard. ODM is:

- Used in industry, academia, and government mapping
- Validated specifically on the Helenenschacht dataset (the dataset is in
  ODM's own test repo)
- Free, open-source (AGPL), runs locally via Docker
- Produces the standard set of deliverables: orthomosaic (GeoTIFF + PNG),
  DSM, DTM, dense point cloud (LAZ + PLY), textured 3D mesh, and a
  processing report

The `pipeline.py` script wraps ODM with:

- Pre-flight validation (image readability, sharpness via Laplacian
  variance, size sanity)
- Docker invocation with parameters tuned for thesis-quality results
- Output collection from ODM's nested project structure into a flat
  `deliverables/` folder
- Thesis-ready visualizations: orthomosaic figure, DSM figure with
  proper terrain colormap and elevation legend, camera-positions plot
- A `RUN_REPORT.md` summarizing inputs, registration rate, and outputs

## Honest limitations

I am telling you these so you can plan around them, not after the fact:

1. **Requires Docker.** If your machine cannot run Docker (very old hardware,
   restricted institutional laptop), this pipeline will not run. There is no
   reliable pure-Python alternative for production photogrammetry. If Docker
   is impossible, the alternatives are: a Linux machine with native ODM, or
   the WebODM cloud service (`https://webodm.net/`).

2. **First run downloads ~3 GB.** The ODM Docker image is large.

3. **Runtime on 176 images:** 30–90 min on a typical laptop CPU. Fast mode
   (`--fast` flag) drops it to ~20 min at lower point cloud density.

4. **Memory:** ODM needs at least 8 GB RAM for 176 images at full quality.
   On 8 GB systems use `--fast`. On 16 GB+ systems use full quality.

5. **Helenenschacht is a forest dataset.** Even ODM will produce holes in
   dense areas of unbroken canopy because there's no parallax there. This is
   a property of the data, not a pipeline bug. Expect the orthomosaic to look
   excellent on roads/clearings and have soft texture in dense forest.

## File layout

```
.
├── pipeline.py          # the actual pipeline (run this)
├── requirements.txt     # pip install -r requirements.txt
└── README.md            # this file
```

After running, the output folder contains:

```
your_output_folder/
├── odm_project/                  # raw ODM working directory (large)
├── deliverables/                 # the actual results, flat layout
│   ├── orthomosaic.tif           # georeferenced orthomosaic (GeoTIFF)
│   ├── orthomosaic.png           # same, PNG for thesis figures
│   ├── dsm.tif                   # digital surface model (geo-tiff, float)
│   ├── dtm.tif                   # digital terrain model
│   ├── dense_cloud.laz           # LAS/LAZ point cloud (open in CloudCompare)
│   ├── dense_cloud.ply           # same, PLY for MeshLab
│   ├── mesh.obj                  # textured 3D mesh
│   ├── camera_poses.geojson      # camera positions in WGS84
│   ├── cameras.json              # camera intrinsics
│   └── odm_report.pdf            # ODM's own processing report
├── thesis_figures/
│   ├── fig_orthomosaic.png
│   ├── fig_dsm.png
│   └── fig_camera_poses.png
├── odm_log.txt                   # full ODM stdout/stderr
└── RUN_REPORT.md                 # summary of this run
```

## Setup

### 1. Install Docker Desktop

- Windows / Mac: download from <https://www.docker.com/products/docker-desktop/>
- Linux: `sudo apt install docker.io` (Debian/Ubuntu) and add yourself to the
  `docker` group: `sudo usermod -aG docker $USER`, then log out/in.

After install, **start Docker Desktop** and confirm it's running. The whale
icon should be in your tray/menu bar.

### 2. Install Python dependencies

```
pip install -r requirements.txt
```

These are only used for image validation and figure generation. The
reconstruction is done inside the Docker container.

### 3. Pull the ODM image (optional — pipeline does this automatically)

```
docker pull opendronemap/odm:latest
```

This is ~3 GB and only needs to be done once.

## Run

```
python pipeline.py --images /path/to/your/photos --output /path/to/results
```

For a faster test run at reduced quality:

```
python pipeline.py --images /path/to/photos --output /path/to/results --fast
```

The script prints progress at each ODM stage. Full ODM output is in
`results/odm_log.txt`.

## Validation

After the pipeline completes, check `RUN_REPORT.md`. Indicators of success:

- **Cameras registered ≥ 85%** of input images. For Helenenschacht's 176
  images, expect 160–175 registered.
- **`orthomosaic.tif` exists and is several hundred MB.** A few-MB
  orthomosaic means most images failed to register.
- **`dsm.tif` exists and elevation values are physically plausible** (for
  Helenenschacht: a few hundred meters above sea level, range of tens of
  meters across the scene). Open in QGIS or read with `gdalinfo dsm.tif`.
- **Open `dense_cloud.laz` in CloudCompare** — you should see a recognizable
  3D forest with structured trees, not a noise blob.

Indicators of trouble:

- Cameras registered < 50%: too little overlap, or motion blur, or very
  few features (very dense canopy with no edges). Solution: provide more
  overlapping images, or accept the partial reconstruction.
- ODM exits with error in log: usually a Docker memory limit. Increase
  Docker Desktop's RAM allocation (Settings → Resources → Memory) to
  ≥ 8 GB.
- Orthomosaic has black wedges: cameras in those areas didn't register.
  This is a data-coverage issue, not a pipeline bug.

## Failure → fix mapping (vs the previous pipeline)

| Previous failure | Root cause | Fix in this pipeline |
|---|---|---|
| 78/176 cameras stacked on a line | Single-pair BA, no global optimization | ODM uses Ceres-based incremental BA over all cameras |
| Dense cloud spans ±200,000 units | Wrong K + degenerate 2-view stereo | ODM reads EXIF camera params and does proper PatchMatch MVS |
| DSM has starburst pattern | griddata over outlier-contaminated cloud | ODM grids only the inlier-filtered georeferenced cloud |
| Orthomosaic is broken fragments | `cv2.Stitcher` is wrong algorithm class | ODM does proper DSM-based orthorectification per pixel |
| OOM on 176 images | All-pairs matching + RAM-resident point list | ODM uses GPS-based neighbor selection and streams |

## What to put in your thesis

The methodology section honestly states:

> The reconstruction was performed using OpenDroneMap (ODM), an open-source
> photogrammetry toolchain implementing OpenSfM for Structure-from-Motion
> and OpenMVS for dense reconstruction. The pipeline was driven by a custom
> Python wrapper that handles input validation, Docker invocation, output
> collection, and figure generation.

This is a perfectly reasonable academic position. A thesis that *integrates*
mature tools into a validated, reproducible workflow is a stronger contribution
than a from-scratch SfM that doesn't actually work. State the components, show
the results, document the failure modes you tested.
