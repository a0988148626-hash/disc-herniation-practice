"""Geometric plausibility check of the disc_height_mm (tilt-corrected true
thickness) method on an INDEPENDENT external dataset: the TUM/OSF
multi-scanner lumbar vertebral body + disc segmentation database
(osf.io/qx5rt, CC BY 4.0, 34 patients, no herniation labels).

This is NOT a classification/AUC validation (no ground-truth herniation
label exists in this dataset) - it only checks whether the SAME geometric
method, applied to a completely different population/scanners/hospital/
label format, produces anatomically plausible numbers (disc thickness in a
sane range, decreasing from lower to upper lumbar levels) and recovers a
similar tilt-angle pattern to what was found in SPIDER.

Differences from the SPIDER pipeline that make this NOT a drop-in reuse:
- Each disc level's mask is its own separate file (BS_L1_2.nii.gz, etc,
  "BS" = Bandscheibe/disc), not encoded as a multi-label volume.
- Voxel values are float and noisy near zero (resampling artifacts) rather
  than clean integer labels - thresholded at > 0.5 here.
- The direction cosine matrix is NOT identity (real clinical positioning,
  unlike SPIDER's apparently preprocessed-to-identity volumes), so voxel
  index deltas scaled by spacing are NOT valid physical coordinates here.
  Full affine (origin + direction @ (spacing * index)) is used instead,
  and PCA is done directly in 3D physical space on the whole disc mask
  (more robust than SPIDER's per-2D-sagittal-slice approach, which only
  happened to be valid there because that dataset's direction was exactly
  identity).
- No L5/S1 disc is labeled in this dataset (only L1-L2 through L4-L5), so
  the specific "L5/S1 more tilted" finding from SPIDER cannot be directly
  re-checked here, only the general decreasing-thickness-by-level pattern.

Usage:
    python plausibility_check.py <tum_raw_download_dir> <output_csv>
"""
import sys
import csv
import glob
import os

import numpy as np
import SimpleITK as sitk


def disc_true_thickness_and_tilt(mask_path, threshold=0.5, min_points=20):
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)  # (iz, iy, ix) numpy order = reverse of ITK (ix,iy,iz)
    mask = arr > threshold
    if mask.sum() < min_points:
        return None

    iz, iy, ix = np.where(mask)  # numpy array indices
    # Convert to ITK index order (ix, iy, iz) for the physical transform
    idx = np.stack([ix, iy, iz], axis=1).astype(float)

    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())
    direction = np.array(img.GetDirection()).reshape(3, 3)

    phys = origin + idx * spacing @ direction.T  # (N,3) physical mm points
    # equivalent to: for each point, origin + direction @ (spacing*idx)

    phys_centered = phys - phys.mean(axis=0)
    cov = np.cov(phys_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    minor_axis = eigvecs[:, 0]  # smallest-variance direction = disc plane normal

    proj_minor = phys_centered @ minor_axis
    thickness_mm = proj_minor.max() - proj_minor.min()

    # Tilt angle: angle between the disc-plane normal and the physical
    # Superior axis. LPS convention: physical +z = Superior.
    world_superior = np.array([0.0, 0.0, 1.0])
    cos_angle = abs(np.dot(minor_axis, world_superior))
    tilt_from_superior_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    # A disc plane that is horizontal (normal points along S-I) gives a
    # SMALL angle here; report the complement so it reads the same way as
    # the SPIDER script's "angle between disc plane and S-I axis" (larger
    # = more horizontal/normal, smaller = more tilted).
    tilt_angle_deg = 90.0 - tilt_from_superior_deg

    return float(thickness_mm), float(tilt_angle_deg), int(mask.sum())


def main(raw_dir, output_csv):
    rows = []
    pattern = os.path.join(raw_dir, "**", "Labels", "BS_*.nii.gz")
    for path in sorted(glob.glob(pattern, recursive=True)):
        parts = path.replace("\\", "/").split("/")
        patient = parts[-4]
        scanner = parts[-3]
        level = os.path.basename(path).replace("BS_", "").replace(".nii.gz", "")
        result = disc_true_thickness_and_tilt(path)
        if result is None:
            print(f"skip (too few voxels / empty): {path}")
            continue
        thickness_mm, tilt_angle_deg, n_voxels = result
        rows.append({
            "patient": patient,
            "scanner": scanner,
            "disc_level": level,
            "thickness_mm": thickness_mm,
            "tilt_angle_deg": tilt_angle_deg,
            "n_voxels": n_voxels,
        })
        print(f"{patient}/{scanner}/{level}: thickness={thickness_mm:.2f}mm tilt={tilt_angle_deg:.1f}deg")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} disc-level rows to {output_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
