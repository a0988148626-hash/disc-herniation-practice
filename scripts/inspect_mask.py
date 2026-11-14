"""Inspect a single SPIDER mask .mha file: print unique label values, shape, spacing."""
import sys
import SimpleITK as sitk
import numpy as np

def main(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    print("path:", path)
    print("size (x,y,z):", img.GetSize())
    print("spacing (x,y,z):", img.GetSpacing())
    print("origin:", img.GetOrigin())
    print("direction:", img.GetDirection())
    print("array shape (z,y,x):", arr.shape)
    print("dtype:", arr.dtype)
    vals, counts = np.unique(arr, return_counts=True)
    print("unique labels and voxel counts:")
    for v, c in zip(vals, counts):
        print(f"  {v}: {c}")

if __name__ == "__main__":
    main(sys.argv[1])
