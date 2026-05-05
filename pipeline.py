"""
UAV Aerial Mapping Pipeline — Production Build
================================================
Replaces the broken pure-OpenCV reconstruction with OpenDroneMap (ODM),
the open-source photogrammetry standard used in industry and academia.

PIPELINE:
  Photos → ODM (SfM + MVS + DSM + Ortho) → Validation → Thesis-ready outputs

Requires:
  - Docker Desktop installed and running
  - 8 GB RAM minimum (16 GB recommended for 176 images)
  - 20 GB free disk space
  - Python 3.9+

Run:
  python pipeline.py --images /path/to/photos --output /path/to/results

Author: rebuilt for graduation thesis
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# 0.  DEPENDENCY & ENVIRONMENT CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_docker():
    """Verify Docker is installed and the daemon is running."""
    try:
        r = subprocess.run(["docker", "--version"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return False, "Docker command failed"
        version = r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, "Docker not installed or not on PATH"

    try:
        r = subprocess.run(["docker", "info"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return False, "Docker daemon not running (start Docker Desktop)"
    except subprocess.TimeoutExpired:
        return False, "Docker daemon unresponsive"

    return True, version


def check_image_folder(folder):
    """Validate the input image folder exists and has enough usable images."""
    p = Path(folder)
    if not p.is_dir():
        return False, f"Folder not found: {folder}", []

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}
    imgs = sorted([f for f in p.iterdir() if f.suffix in exts])

    if len(imgs) < 5:
        return False, f"Only {len(imgs)} images found — need at least 5", imgs

    # Read first image to validate
    test = cv2.imread(str(imgs[0]))
    if test is None:
        return False, f"Cannot read {imgs[0].name} — corrupted?", imgs
    h, w = test.shape[:2]

    return True, f"{len(imgs)} images, first is {w}×{h}", imgs


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PRE-FLIGHT IMAGE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate_images(image_paths):
    """
    Pre-check that images are usable for SfM:
      - readable
      - non-trivial size (>= 640 px on long edge)
      - sufficient texture (Laplacian variance > threshold = sharp enough)
      - have EXIF (helpful but not required for ODM)
    """
    print("\n  Validating images for SfM suitability...")

    issues = []
    sharpness_scores = []
    sizes = []

    for i, p in enumerate(image_paths):
        img = cv2.imread(str(p))
        if img is None:
            issues.append(f"  ❌ {p.name}: unreadable")
            continue
        h, w = img.shape[:2]
        sizes.append((w, h))
        if max(w, h) < 640:
            issues.append(f"  ⚠️  {p.name}: too small ({w}×{h})")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Laplacian variance — standard sharpness proxy
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_scores.append(sharpness)

    sharpness_scores = np.array(sharpness_scores)
    median_sharp = float(np.median(sharpness_scores))
    blurry_threshold = median_sharp * 0.4  # 40% of median is "blurry"
    blurry_count = int((sharpness_scores < blurry_threshold).sum())

    print(f"    Median sharpness (Laplacian var): {median_sharp:.1f}")
    print(f"    Likely-blurry images (<40% of median): {blurry_count}/{len(image_paths)}")
    print(f"    Image size range: {min(sizes)} to {max(sizes)}")

    if issues:
        print(f"    {len(issues)} issues:")
        for s in issues[:10]:
            print(s)
        if len(issues) > 10:
            print(f"    ... and {len(issues)-10} more")

    if blurry_count > len(image_paths) * 0.3:
        print("    ⚠️  More than 30% of images appear blurry — results may suffer.")
    else:
        print("    ✅ Image set is suitable for SfM.")

    return {
        "n_images": len(image_paths),
        "median_sharpness": median_sharp,
        "blurry_count": blurry_count,
        "size_range": [min(sizes), max(sizes)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  RUN OPENDRONEMAP (the actual reconstruction engine)
# ─────────────────────────────────────────────────────────────────────────────
def run_odm(images_dir, output_dir, fast_mode=False):
    """
    Run OpenDroneMap via its official Docker image.

    ODM does:
      1. EXIF parsing + camera model selection
      2. SIFT feature extraction (or HAHOG if SIFT unavailable)
      3. Brute-force matching with geometric pre-filtering by GPS
      4. Incremental SfM with proper bundle adjustment (Ceres Solver)
      5. Multi-view stereo via OpenSfM/OpenMVS
      6. Mesh reconstruction
      7. DSM/DEM generation via point cloud gridding
      8. Orthomosaic via per-pixel DSM-based orthorectification
    """
    images_dir = Path(images_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ODM's expected layout: project_root/code/images/*.jpg
    project_root = output_dir / "odm_project"
    odm_images = project_root / "images"
    odm_images.mkdir(parents=True, exist_ok=True)

    # Copy/symlink images into project structure
    print(f"\n  Preparing ODM project at {project_root}")
    n_copied = 0
    for img in images_dir.iterdir():
        if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            dest = odm_images / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
            n_copied += 1
    print(f"    {n_copied} images staged for ODM")

    # Volume mount differs between Windows and Unix
    project_str = str(project_root)
    if platform.system() == "Windows":
        project_str = project_str.replace("\\", "/")
        if project_str[1] == ":":
            project_str = "/" + project_str[0].lower() + project_str[2:]

    # ODM command — these flags are tuned for thesis-quality results without
    # GPU and with reasonable RAM. fast_mode trades quality for speed.
    odm_args = [
        "--feature-quality", "medium" if fast_mode else "high",
        "--matcher-neighbors", "8",
        "--matcher-type", "flann",
        "--min-num-features", "10000",
        "--pc-quality", "medium" if fast_mode else "high",
        "--mesh-octree-depth", "10" if fast_mode else "11",
        "--orthophoto-resolution", "5",  # cm/pixel
        "--dsm",
        "--dtm",
        "--cog",
        "--use-3dmesh",
    ]
    if fast_mode:
        odm_args += ["--fast-orthophoto"]

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{project_str}:/datasets/code",
        "opendronemap/odm:latest",
        "--project-path", "/datasets",
        "code",  # the project name inside /datasets
    ] + odm_args

    print("\n  Pulling/launching ODM Docker image (first run downloads ~3 GB)...")
    print("  Command:", " ".join(cmd))
    print("  This will take 20–90 minutes depending on hardware. Streaming logs:\n")
    print("  " + "─" * 70)

    log_path = output_dir / "odm_log.txt"
    t_start = time.time()

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            # Surface only the milestone lines to the console
            stripped = line.strip()
            if any(m in stripped for m in [
                "Running",
                "Reconstruction will use",
                "Detecting",
                "Matching",
                "Found",
                "ERROR",
                "Triangulating",
                "Generating",
                "stage",
                "Total reconstruction",
                "Cameras",
                "points",
            ]):
                print(f"  │ {stripped[:120]}")
        proc.wait()

    elapsed = time.time() - t_start
    print("  " + "─" * 70)
    print(f"  ODM finished in {elapsed/60:.1f} min, exit code {proc.returncode}")

    if proc.returncode != 0:
        print(f"  ❌ ODM failed. Full log: {log_path}")
        return False, project_root

    return True, project_root


# ─────────────────────────────────────────────────────────────────────────────
# 3.  COLLECT & VALIDATE ODM OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
def collect_outputs(project_root, output_dir):
    """
    Pull ODM's key deliverables out of its nested output structure into a
    flat folder for the thesis.
    """
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    deliverables = output_dir / "deliverables"
    deliverables.mkdir(exist_ok=True)

    # Map ODM internal paths → flat thesis paths
    mapping = {
        "odm_orthophoto/odm_orthophoto.tif":   "orthomosaic.tif",
        "odm_orthophoto/odm_orthophoto.png":   "orthomosaic.png",
        "odm_dem/dsm.tif":                     "dsm.tif",
        "odm_dem/dtm.tif":                     "dtm.tif",
        "odm_georeferencing/odm_georeferenced_model.laz":  "dense_cloud.laz",
        "odm_georeferencing/odm_georeferenced_model.ply":  "dense_cloud.ply",
        "odm_texturing/odm_textured_model_geo.obj":        "mesh.obj",
        "odm_report/report.pdf":               "odm_report.pdf",
        "cameras.json":                        "cameras.json",
        "shots.geojson":                       "camera_poses.geojson",
    }

    found, missing = [], []
    for src_rel, dst_name in mapping.items():
        src = project_root / src_rel
        if src.exists():
            dst = deliverables / dst_name
            shutil.copy2(src, dst)
            size_mb = dst.stat().st_size / 1e6
            found.append((dst_name, size_mb))
        else:
            missing.append(src_rel)

    return found, missing, deliverables


# ─────────────────────────────────────────────────────────────────────────────
# 4.  THESIS-READY VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
def make_visualizations(deliverables_dir, output_dir):
    """Generate the figures you'll put in your thesis."""
    deliverables_dir = Path(deliverables_dir)
    output_dir = Path(output_dir)
    figs_dir = output_dir / "thesis_figures"
    figs_dir.mkdir(exist_ok=True)

    # ── Orthomosaic ──────────────────────────────────────────────────────────
    ortho_path = deliverables_dir / "orthomosaic.png"
    if not ortho_path.exists():
        ortho_tif = deliverables_dir / "orthomosaic.tif"
        if ortho_tif.exists():
            img = cv2.imread(str(ortho_tif), cv2.IMREAD_UNCHANGED)
            if img is not None:
                cv2.imwrite(str(ortho_path), img)

    if ortho_path.exists():
        ortho = cv2.imread(str(ortho_path))
        if ortho is not None:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.imshow(cv2.cvtColor(ortho, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Orthomosaic — {ortho.shape[1]}×{ortho.shape[0]} px",
                         fontsize=12, fontweight="bold")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(figs_dir / "fig_orthomosaic.png",
                        dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  ✅ Orthomosaic figure → fig_orthomosaic.png")

    # ── DSM ──────────────────────────────────────────────────────────────────
    dsm_path = deliverables_dir / "dsm.tif"
    if dsm_path.exists():
        try:
            # GeoTIFF readable via OpenCV with IMREAD_UNCHANGED
            dsm = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
            if dsm is not None and dsm.dtype != np.uint8:
                # Mask invalid (ODM uses -9999 or NaN)
                valid = (dsm > -1000) & np.isfinite(dsm)
                vmin, vmax = np.percentile(dsm[valid], [2, 98])

                fig, ax = plt.subplots(figsize=(12, 9))
                masked = np.ma.array(dsm, mask=~valid)
                im = ax.imshow(masked, cmap="terrain", vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, label="Elevation (m)")
                ax.set_title("Digital Surface Model (DSM)",
                             fontsize=12, fontweight="bold")
                ax.axis("off")
                plt.tight_layout()
                plt.savefig(figs_dir / "fig_dsm.png",
                            dpi=200, bbox_inches="tight")
                plt.close()
                print(f"  ✅ DSM figure → fig_dsm.png")
        except Exception as e:
            print(f"  ⚠️  DSM rendering failed: {e}")

    # ── Camera poses ─────────────────────────────────────────────────────────
    poses_geo = deliverables_dir / "camera_poses.geojson"
    cams_json = deliverables_dir / "cameras.json"

    n_cameras = None
    if poses_geo.exists():
        with open(poses_geo) as f:
            data = json.load(f)
        feats = data.get("features", [])
        n_cameras = len(feats)

        # Plot top-down camera positions
        if feats:
            xs, ys = [], []
            for ft in feats:
                geom = ft.get("geometry", {})
                if geom.get("type") == "Point":
                    x, y = geom["coordinates"][:2]
                    xs.append(x); ys.append(y)

            if xs:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(xs, ys, s=30, c="red", marker="^", edgecolors="k")
                ax.set_title(f"Reconstructed camera positions  (n = {len(xs)})",
                             fontsize=12, fontweight="bold")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_aspect("equal", adjustable="datalim")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(figs_dir / "fig_camera_poses.png",
                            dpi=180, bbox_inches="tight")
                plt.close()
                print(f"  ✅ Camera poses figure → fig_camera_poses.png  "
                      f"({len(xs)} cameras)")

    return n_cameras, figs_dir


# ─────────────────────────────────────────────────────────────────────────────
# 5.  REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_report(output_dir, validation_info, n_cameras, found, missing, elapsed_min):
    output_dir = Path(output_dir)
    report_path = output_dir / "RUN_REPORT.md"

    with open(report_path, "w") as f:
        f.write("# Pipeline Run Report\n\n")
        f.write(f"**Total runtime:** {elapsed_min:.1f} min\n\n")

        f.write("## Input validation\n\n")
        f.write(f"- Images: {validation_info['n_images']}\n")
        f.write(f"- Median sharpness: {validation_info['median_sharpness']:.1f}\n")
        f.write(f"- Blurry images: {validation_info['blurry_count']}\n")
        f.write(f"- Size range: {validation_info['size_range']}\n\n")

        f.write("## Reconstruction\n\n")
        if n_cameras is not None:
            ratio = n_cameras / validation_info["n_images"]
            f.write(f"- Cameras registered: {n_cameras} / "
                    f"{validation_info['n_images']} ({ratio:.0%})\n")
            if ratio >= 0.85:
                f.write(f"- Status: ✅ Excellent registration rate\n")
            elif ratio >= 0.6:
                f.write(f"- Status: ⚠️ Acceptable, "
                        f"but consider more overlap next time\n")
            else:
                f.write(f"- Status: ❌ Poor registration — "
                        f"check overlap and image quality\n")
        f.write("\n")

        f.write("## Deliverables produced\n\n")
        for name, size in found:
            f.write(f"- ✅ `{name}`  ({size:.1f} MB)\n")
        if missing:
            f.write("\n## Missing outputs\n\n")
            for m in missing:
                f.write(f"- ⚠️ `{m}`\n")

    print(f"\n  Report written → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="UAV mapping pipeline — production build (ODM-based)")
    parser.add_argument("--images", required=True, help="Folder of input images")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--fast", action="store_true",
                        help="Faster, lower-quality run (for testing)")
    args = parser.parse_args()

    print("═" * 72)
    print(" UAV AERIAL MAPPING PIPELINE — PRODUCTION BUILD")
    print("═" * 72)

    t_total_start = time.time()

    # Step 0 — environment
    print("\n[0/5] Environment check")
    ok, msg = check_docker()
    if not ok:
        print(f"  ❌ Docker check failed: {msg}")
        print("\n  Install Docker Desktop from https://docker.com/products/docker-desktop")
        print("  Then start it and re-run this script.")
        sys.exit(1)
    print(f"  ✅ Docker OK ({msg})")

    # Step 1 — image folder
    print("\n[1/5] Image folder check")
    ok, msg, imgs = check_image_folder(args.images)
    if not ok:
        print(f"  ❌ {msg}")
        sys.exit(1)
    print(f"  ✅ {msg}")
    validation = validate_images(imgs)

    # Step 2 — run ODM
    print("\n[2/5] Running OpenDroneMap")
    ok, project_root = run_odm(args.images, args.output, fast_mode=args.fast)
    if not ok:
        print("\n  ❌ ODM run failed. See odm_log.txt for details.")
        sys.exit(2)

    # Step 3 — collect outputs
    print("\n[3/5] Collecting deliverables")
    found, missing, deliverables = collect_outputs(project_root, args.output)
    if not found:
        print("  ❌ ODM produced no outputs — reconstruction failed.")
        sys.exit(3)
    for name, size in found:
        print(f"  ✅ {name:<32} {size:>8.1f} MB")
    for m in missing[:5]:
        print(f"  ⚠️  missing: {m}")

    # Step 4 — visualizations
    print("\n[4/5] Generating thesis figures")
    n_cameras, figs_dir = make_visualizations(deliverables, args.output)

    # Step 5 — report
    print("\n[5/5] Writing report")
    elapsed_min = (time.time() - t_total_start) / 60
    write_report(args.output, validation, n_cameras, found, missing, elapsed_min)

    print("\n" + "═" * 72)
    print(" PIPELINE COMPLETE")
    print("═" * 72)
    print(f"  Total time         : {elapsed_min:.1f} min")
    print(f"  Cameras registered : {n_cameras}")
    print(f"  Deliverables       : {deliverables}")
    print(f"  Thesis figures     : {figs_dir}")
    print(f"  Report             : {Path(args.output) / 'RUN_REPORT.md'}")


if __name__ == "__main__":
    main()
