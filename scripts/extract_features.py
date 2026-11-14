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

Vertebral body vs. posterior-element separation
-------------------------------------------------
Each sagittal slice (fixed x, 2D plane over z,y) is a real anatomical image
plane. Within such a slice, the vertebral body and the posterior elements
(pedicle/lamina/spinous process) are usually two SEPARATE connected
components of the "vertebra" label, because the spinal canal sits between
them. We run 2D connected-component labeling (8-connectivity) per sagittal
slice and classify each component as "body" (anterior) or "posterior
element" by majority vote of its pixels against the spinal canal's local
y-position at the matching z rows. Known failure mode: on slices that pass
through the pedicle, body and posterior elements are physically fused into
one component; our majority-vote still assigns the whole fused blob to one
class, which can bias the body's posterior border slightly more posterior
than the true cortical margin on those particular slices (documented as a
limitation in the README, not hidden).

Usage:
    python extract_features.py <masks_dir> <overview_csv> <gradings_csv> <output_csv>
"""
import sys
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


VERTEBRA_LABELS = set(range(1, 26)) | set(range(101, 126))
DISC_LABEL_BASE = 200
CC_STRUCTURE = np.ones((3, 3))  # 8-connectivity for 2D connected components


MAX_SAGITTAL_SLICE_COUNT = 200  # see is_sagittal_multislice() docstring


def is_sagittal_multislice(path: Path) -> bool:
    """Check that this .mha file actually has the geometry this pipeline
    assumes: a 2D multi-slice sagittal stack, i.e. few (sparse) slices along
    the left-right axis (ITK size[0]) and a much larger in-plane extent.

    Discovered during development: not every file in this dataset is such a
    stack. Some "_t1"/"_t2" files are actually axial reformats with a square
    512x512 in-plane FOV and only ~17-25 slices along the superior-inferior
    axis instead (e.g. patient 161's t2 series: ITK size (512, 512, 19)), and
    every "_t2_SPACE"/"_t1_SPACE" file is a near-isotropic 3D acquisition
    with size[0]=512 as well. Running the sagittal-slice-by-slice geometric
    logic on these produces meaningless results (this is what caused the
    disc_height_mm outlier such as patient 161 level 1's ~60mm "height" -
    not simple noise, but a genuine orientation mismatch that a median
    instead of max/min over z does NOT fix). Across the whole dataset,
    ITK size[0] is cleanly bimodal (<=144 for real sagittal stacks, 512 for
    these atypical files), so a threshold of 200 separates them with no
    ambiguous cases.
    """
    img = sitk.ReadImage(str(path))
    return img.GetSize()[0] < MAX_SAGITTAL_SLICE_COUNT


def pick_file_for_patient(masks_dir: Path, patient_id: str):
    """Prefer t2, then t1, then t2_SPACE, skipping any candidate whose
    geometry is not a standard sagittal multi-slice stack (see
    is_sagittal_multislice). A patient with no sequence passing this check
    is reported as excluded rather than silently processed with
    geometrically-invalid assumptions."""
    for suffix in ("t2", "t1", "t2_SPACE", "t1_SPACE"):
        candidate = masks_dir / f"{patient_id}_{suffix}.mha"
        if candidate.exists() and is_sagittal_multislice(candidate):
            return candidate, suffix
    return None, None


def load_mask_array(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    sx, sy, sz = img.GetSpacing()  # spacing in (x, y, z) order
    return arr, (sz, sy, sx)  # return spacing matching array axis order (z,y,x)


def _classify_components_as_body(labeled, n_components, canal_slice2d):
    """labeled: 2D int array (z,y) of connected-component ids (0 = background).
    canal_slice2d: 2D boolean array (z,y) for the spinal canal in the same slice.
    Returns the set of component ids classified as vertebral BODY (anterior to
    the canal), using a per-component majority vote against the canal's local
    y-position at each component pixel's own z row."""
    n_z = labeled.shape[0]
    canal_y_per_z = np.full(n_z, np.nan)
    for z in range(n_z):
        ys = np.where(canal_slice2d[z])[0]
        if ys.size:
            canal_y_per_z[z] = ys.mean()
    valid_canal = ~np.isnan(canal_y_per_z)
    canal_overall_mean = canal_y_per_z[valid_canal].mean() if valid_canal.any() else None

    body_components = set()
    comp_mean_y = {}
    for comp_id in range(1, n_components + 1):
        zz, yy = np.where(labeled == comp_id)
        comp_mean_y[comp_id] = yy.mean()
        local_canal = canal_y_per_z[zz]
        has_local = ~np.isnan(local_canal)
        if has_local.sum() >= max(3, int(0.3 * len(zz))):
            is_body = (yy[has_local] < local_canal[has_local]).mean() >= 0.5
        elif canal_overall_mean is not None:
            is_body = yy.mean() < canal_overall_mean
        else:
            is_body = True  # no canal reference anywhere in this slice
        if is_body:
            body_components.add(comp_id)

    if not body_components and comp_mean_y:
        # Degenerate case (e.g. canal mask entirely absent in this slice):
        # fall back to the single most-anterior component.
        body_components = {min(comp_mean_y, key=comp_mean_y.get)}
    return body_components


def compute_vertebral_body_border_maps(arr):
    """Compute, once per volume, 2D maps (z, x) of the vertebral BODY's
    posterior-most and anterior-most y index, using per-sagittal-slice (fixed
    x) connected-component analysis to separate body from posterior elements.
    Returns (body_post_y, body_ant_y), both shape (n_z, n_x), np.nan where
    no vertebral body voxel could be classified."""
    n_z, n_y, n_x = arr.shape
    vert_mask = np.isin(arr, list(VERTEBRA_LABELS))
    canal_mask = arr == 100

    body_post_y = np.full((n_z, n_x), np.nan)
    body_ant_y = np.full((n_z, n_x), np.nan)

    for x in range(n_x):
        vert_slice2d = vert_mask[:, :, x]  # (z, y)
        if not vert_slice2d.any():
            continue
        canal_slice2d = canal_mask[:, :, x]
        labeled, n_components = ndimage.label(vert_slice2d, structure=CC_STRUCTURE)
        if n_components == 0:
            continue
        body_components = _classify_components_as_body(labeled, n_components, canal_slice2d)
        body_mask2d = np.isin(labeled, list(body_components))
        for z in range(n_z):
            ys = np.where(body_mask2d[z])[0]
            if ys.size:
                body_post_y[z, x] = ys.max()
                body_ant_y[z, x] = ys.min()

    return body_post_y, body_ant_y


def compute_disc_height_mm_raw_zextent(disc_mask, sz):
    """Disc height (v3, kept only for comparison): median, across sagittal
    (x) columns, of each column's LOCAL z-extent (vertical S-I bounding
    box), using only the largest contiguous run of z within that column.
    This is systematically biased upward at tilted disc levels (see
    compute_disc_true_thickness_mm / README "L5/S1 離群值" discussion):
    a disc plane tilted by angle theta from horizontal has its vertical
    (S-I) extent stretched by roughly 1/cos(theta) relative to its true
    perpendicular thickness."""
    n_x = disc_mask.shape[2]
    heights_vox = []
    for x in range(n_x):
        zs = np.where(disc_mask[:, :, x].any(axis=1))[0]
        if zs.size == 0:
            continue
        gaps = np.where(np.diff(zs) > 1)[0]
        runs = np.split(zs, gaps + 1)
        longest = max(runs, key=len)
        heights_vox.append(longest.max() - longest.min() + 1)
    if not heights_vox:
        return np.nan
    return float(np.median(heights_vox)) * sz


def compute_disc_true_thickness_mm(disc_mask, sy, sz, min_points=6):
    """Disc height (v4): tilt-corrected true thickness.

    A lumbar disc is a thin slab; in a sagittal (z,y) cross-section it shows
    up as an elongated blob whose long axis runs roughly along the disc's
    own plane (tilted by the local lumbosacral/segmental angle, not
    necessarily horizontal) and whose SHORT axis is the disc's true
    thickness, measured perpendicular to its own plane rather than along the
    image's vertical (S-I) axis.

    For each sagittal (x) slice, fit PCA (via eigendecomposition of the
    covariance matrix) to the disc mask's point cloud, in physical mm
    (voxel indices scaled by spacing - the two in-plane axes have different
    voxel spacing, so this step matters). The minor eigenvector gives the
    thickness direction; projecting the points onto it and taking the
    max-min range gives the true thickness for that slice, and the major
    eigenvector's angle relative to the z (S-I) axis gives the local tilt
    angle. Returns (median true thickness in mm, median tilt angle in
    degrees) across the slices with enough points for a stable fit.
    """
    n_x = disc_mask.shape[2]
    thickness_vals = []
    tilt_angles_deg = []
    for x in range(n_x):
        zz, yy = np.where(disc_mask[:, :, x])
        if zz.size < min_points:
            continue
        pts = np.stack([zz * sz, yy * sy], axis=1)  # physical mm: (z_mm, y_mm)
        pts = pts - pts.mean(axis=0)
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
        minor_axis = eigvecs[:, 0]
        major_axis = eigvecs[:, 1]
        proj_minor = pts @ minor_axis
        thickness_vals.append(proj_minor.max() - proj_minor.min())
        tilt_angles_deg.append(np.degrees(np.arctan2(abs(major_axis[1]), abs(major_axis[0]) + 1e-12)))
    if not thickness_vals:
        return np.nan, np.nan
    return float(np.median(thickness_vals)), float(np.median(tilt_angles_deg))


def extract_disc_features(arr, spacing_zyx, disc_label, body_post_y, body_ant_y):
    """arr: (z,y,x) label volume. spacing_zyx: (sz, sy, sx) mm/voxel.
    body_post_y / body_ant_y: precomputed (n_z, n_x) vertebral body border
    maps from compute_vertebral_body_border_maps(). Returns a dict of
    features, or None if the disc is not present in this volume."""
    sz, sy, sx = spacing_zyx
    disc_mask = arr == disc_label
    canal_mask = arr == 100

    z_idxs = np.where(disc_mask.any(axis=(1, 2)))[0]
    if z_idxs.size == 0:
        return None

    disc_height_mm_raw_zextent = compute_disc_height_mm_raw_zextent(disc_mask, sz)
    disc_height_mm, disc_tilt_angle_deg = compute_disc_true_thickness_mm(disc_mask, sy, sz)

    bulge_mm_per_xz = []
    canal_encroach_count = 0
    canal_eval_count = 0
    disc_y_extent_per_xz = []
    depths = []

    for z in z_idxs:
        disc_ys_by_x = disc_mask[z]
        canal_slice = canal_mask[z]
        body_post_row = body_post_y[z]
        body_ant_row = body_ant_y[z]
        valid_depth = ~np.isnan(body_post_row) & ~np.isnan(body_ant_row)
        if valid_depth.any():
            depths.append(((body_post_row[valid_depth] - body_ant_row[valid_depth]) * sy).mean())
        for x in range(arr.shape[2]):
            ys = np.where(disc_ys_by_x[:, x])[0]
            if ys.size == 0:
                continue
            disc_post_y = ys.max()
            disc_ant_y = ys.min()
            disc_y_extent_per_xz.append((disc_post_y - disc_ant_y) * sy)
            ref = body_post_row[x]
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

    ap_depth_mm = float(np.mean(depths)) if depths else np.nan
    max_bulge_mm = float(bulge_arr.max())
    mean_bulge_mm = float(bulge_arr.mean())
    canal_encroach_frac = (
        canal_encroach_count / canal_eval_count if canal_eval_count else np.nan
    )

    return {
        "disc_height_mm": disc_height_mm,
        "disc_height_mm_raw_zextent": disc_height_mm_raw_zextent,
        "disc_tilt_angle_deg": disc_tilt_angle_deg,
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
            body_post_y, body_ant_y = compute_vertebral_body_border_maps(arr)
            cache_patient[patient] = (arr, spacing_zyx, seq_used, body_post_y, body_ant_y)

        entry = cache_patient[patient]
        if entry is None:
            missing_patients.append(patient)
            continue
        arr, spacing_zyx, seq_used, body_post_y, body_ant_y = entry

        disc_label = DISC_LABEL_BASE + ivd_label
        feats = extract_disc_features(arr, spacing_zyx, disc_label, body_post_y, body_ant_y)
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
    print(f"Patients with no usable (sagittal multi-slice) mask file: {len(set(missing_patients))}")
    if missing_patients:
        print("  excluded patient ids:", sorted(set(missing_patients), key=int))
    print(f"Disc levels absent from the chosen mask volume: {len(missing_disc_in_mask)}")
    if missing_disc_in_mask[:10]:
        print("  sample:", missing_disc_in_mask[:10])


if __name__ == "__main__":
    main(*sys.argv[1:5])
