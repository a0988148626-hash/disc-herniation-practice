"""Extract simple geometric features for each intervertebral disc (IVD) level
from SPIDER segmentation masks, to be used for a lightweight (non-deep-learning)
disc herniation screening method.

Mask label convention (per SPIDER dataset documentation):
    0         = background
    1-25      = vertebrae, numbered from the bottom (L5 = 1, L4 = 2, ...)
    100       = spinal canal
    101-125   = partially visible vertebrae
    201-225   = intervertebral discs, numbered from the bottom (L5/S1 = 201, ...)

Array axes after SimpleITK GetArrayFromImage(): arr[z, y, x]
    z: superior(+) <-> inferior(-)   (craniocaudal)
    y: posterior(+) <-> anterior(-)  (anteroposterior)
    x: left(+) <-> right(-)          (left-right / sagittal slice index)
(This follows the standard DICOM/ITK LPS convention for an identity direction
matrix, which is what every file in this dataset uses.)

Usage:
    python extract_features.py <masks_dir> <overview_csv> <gradings_csv> <output_csv>
"""
import sys
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk


VERTEBRA_LABELS = set(range(1, 26)) | set(range(101, 126))
DISC_LABEL_BASE = 200


def pick_file_for_patient(masks_dir: Path, patient_id: str):
    """Prefer t2, then t1, then t2_SPACE (document choice as a limitation)."""
    for suffix in ("t2", "t1", "t2_SPACE", "t1_SPACE"):
        candidate = masks_dir / f"{patient_id}_{suffix}.mha"
        if candidate.exists():
            return candidate, suffix
    return None, None


def load_mask_array(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    sx, sy, sz = img.GetSpacing()  # spacing in (x, y, z) order
    return arr, (sz, sy, sx)  # return spacing matching array axis order (z,y,x)


def vertebral_body_posterior_border(arr, canal_label_mask, z_idx):
    """At a given z slice (and across x), estimate the posterior-most (max y)
    voxel of VERTEBRAL BODY (i.e. vertebra voxels anterior to the spinal
    canal), separately per x column. Returns dict x -> body_posterior_y
    (np.nan if not found for that x)."""
    slice_vert = np.isin(arr[z_idx], list(VERTEBRA_LABELS))
    slice_canal = canal_label_mask[z_idx]
    n_x = arr.shape[2]
    result = np.full(n_x, np.nan)
    for x in range(n_x):
        canal_ys = np.where(slice_canal[:, x])[0]
        vert_ys = np.where(slice_vert[:, x])[0]
        if vert_ys.size == 0:
            continue
        if canal_ys.size > 0:
            canal_centroid_y = canal_ys.mean()
            body_ys = vert_ys[vert_ys < canal_centroid_y]
        else:
            # No canal voxel in this column; fall back to all vertebra voxels
            body_ys = vert_ys
        if body_ys.size == 0:
            continue
        result[x] = body_ys.max()
    return result


def vertebral_body_anterior_border(arr, z_idx):
    slice_vert = np.isin(arr[z_idx], list(VERTEBRA_LABELS))
    n_x = arr.shape[2]
    result = np.full(n_x, np.nan)
    for x in range(n_x):
        vert_ys = np.where(slice_vert[:, x])[0]
        if vert_ys.size:
            result[x] = vert_ys.min()
    return result


def extract_disc_features(arr, spacing_zyx, disc_label):
    """arr: (z,y,x) label volume. spacing_zyx: (sz, sy, sx) mm/voxel.
    Returns a dict of features, or None if disc not present in this volume."""
    sz, sy, sx = spacing_zyx
    disc_mask = arr == disc_label
    canal_mask = arr == 100

    z_idxs = np.where(disc_mask.any(axis=(1, 2)))[0]
    if z_idxs.size == 0:
        return None

    z_lo, z_hi = z_idxs.min(), z_idxs.max()
    disc_height_vox = (z_hi - z_lo + 1)
    disc_height_mm = disc_height_vox * sz

    bulge_mm_per_xz = []
    canal_encroach_count = 0
    canal_eval_count = 0
    disc_y_extent_per_xz = []

    for z in z_idxs:
        disc_ys_by_x = disc_mask[z]
        body_post = vertebral_body_posterior_border(arr, canal_mask, z)
        body_ant = vertebral_body_anterior_border(arr, z)
        canal_slice = canal_mask[z]
        for x in range(arr.shape[2]):
            ys = np.where(disc_ys_by_x[:, x])[0]
            if ys.size == 0:
                continue
            disc_post_y = ys.max()
            disc_ant_y = ys.min()
            disc_y_extent_per_xz.append((disc_post_y - disc_ant_y) * sy)
            ref = body_post[x]
            if not np.isnan(ref):
                bulge_mm_per_xz.append((disc_post_y - ref) * sy)
            canal_ys = np.where(canal_slice[:, x])[0]
            if canal_ys.size:
                canal_eval_count += 1
                canal_anterior_y = canal_ys.min()
                if disc_post_y >= canal_anterior_y:
                    canal_encroach_count += 1

    if not bulge_mm_per_xz:
        return None

    bulge_arr = np.array(bulge_mm_per_xz)
    n = len(bulge_arr)
    left_half = bulge_arr[: n // 2]
    right_half = bulge_arr[n // 2:]
    asymmetry_mm = abs(
        (left_half.max() if left_half.size else 0)
        - (right_half.max() if right_half.size else 0)
    )

    # Vertebral body AP depth, averaged over slices where disc is present,
    # used to normalize the bulge measurement to a scale-invariant ratio.
    depths = []
    for z in z_idxs:
        body_post = vertebral_body_posterior_border(arr, canal_mask, z)
        body_ant = vertebral_body_anterior_border(arr, z)
        valid = ~np.isnan(body_post) & ~np.isnan(body_ant)
        if valid.any():
            depths.append(((body_post[valid] - body_ant[valid]) * sy).mean())
    ap_depth_mm = float(np.mean(depths)) if depths else np.nan

    max_bulge_mm = float(bulge_arr.max())
    mean_bulge_mm = float(bulge_arr.mean())
    canal_encroach_frac = (
        canal_encroach_count / canal_eval_count if canal_eval_count else np.nan
    )

    return {
        "disc_height_mm": disc_height_mm,
        "max_bulge_mm": max_bulge_mm,
        "mean_bulge_mm": mean_bulge_mm,
        "bulge_normalized": (max_bulge_mm / ap_depth_mm) if ap_depth_mm else np.nan,
        "asymmetry_mm": float(asymmetry_mm),
        "ap_depth_mm": ap_depth_mm,
        "canal_encroach_frac": canal_encroach_frac,
        "mean_disc_ap_extent_mm": float(np.mean(disc_y_extent_per_xz)),
        "n_xz_slices_with_disc": n,
    }


def main(masks_dir, overview_csv, gradings_csv, output_csv):
    masks_dir = Path(masks_dir)

    # Build patient -> (chosen mask file, sequence used)
    patient_ids = sorted(
        {p.name.split("_")[0] for p in masks_dir.glob("*.mha")},
        key=lambda s: int(s),
    )

    with open(gradings_csv, encoding="utf-8") as f:
        gradings = list(csv.DictReader(f))

    rows_out = []
    missing_patients = []
    missing_disc_in_mask = []

    cache_patient = {}
    for g in gradings:
        patient = g["Patient"]
        ivd_label = int(g["IVD label"])
        if ivd_label <= 0:
            continue  # data artifact rows (see data exploration notes)

        if patient not in cache_patient:
            mask_path, seq_used = pick_file_for_patient(masks_dir, patient)
            if mask_path is None:
                cache_patient[patient] = None
                continue
            arr, spacing_zyx = load_mask_array(mask_path)
            cache_patient[patient] = (arr, spacing_zyx, seq_used)

        entry = cache_patient[patient]
        if entry is None:
            missing_patients.append(patient)
            continue
        arr, spacing_zyx, seq_used = entry

        disc_label = DISC_LABEL_BASE + ivd_label
        feats = extract_disc_features(arr, spacing_zyx, disc_label)
        if feats is None:
            missing_disc_in_mask.append((patient, ivd_label))
            continue

        row = {
            "patient": patient,
            "ivd_label": ivd_label,
            "sequence_used": seq_used,
            "disc_herniation": int(g["Disc herniation"]),
            "disc_bulging": int(g["Disc bulging"]),
            "disc_narrowing": int(g["Disc narrowing"]),
            "pfirrmann_grade": g["Pfirrman grade"],
            "modic": g["Modic"],
        }
        row.update(feats)
        rows_out.append(row)

    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} disc-level feature rows to {output_csv}")
    print(f"Patients with no usable mask file: {len(set(missing_patients))}")
    print(f"Disc levels absent from the chosen mask volume: {len(missing_disc_in_mask)}")
    if missing_disc_in_mask[:10]:
        print("  sample:", missing_disc_in_mask[:10])


if __name__ == "__main__":
    main(*sys.argv[1:5])
