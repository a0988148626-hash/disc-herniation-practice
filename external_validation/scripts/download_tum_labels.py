"""Download ONLY the segmentation label (.nii.gz mask) files from the TUM/OSF
multi-scanner lumbar spine dataset (osf.io/qx5rt, component bj69m, CC BY 4.0).
Deliberately skips the "Images" folders (raw MRI intensities) - not needed
for the geometric plausibility check, and much smaller to download.

Usage:
    python download_tum_labels.py <output_dir>
"""
import sys
import os
import time
import json
import urllib.request

NODE = "bj69m"
BASE = f"https://files.de-1.osf.io/v1/resources/{NODE}/providers/osfstorage"


def api_get(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=40) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)


def list_folder(path_segment=""):
    url = f"{BASE}/{path_segment}"
    return api_get(url)["data"]


def download_file(file_id_path, dest_path, retries=4):
    url = f"{BASE}/{file_id_path}"
    # size>0, not just exists(): a run interrupted mid-write (timeout, network
    # drop) leaves a 0-byte file that exists() would wrongly treat as done,
    # silently skipping it on every rerun (this actually happened once - see
    # the ID07/BS_L1_2.nii.gz incident in the README).
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as f:
                f.write(resp.read())
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3)


def crawl(entries, local_dir, rel_path=""):
    for entry in entries:
        name = entry["attributes"]["name"]
        kind = entry["attributes"]["kind"]
        path = entry["attributes"]["path"].strip("/")
        new_rel = f"{rel_path}/{name}" if rel_path else name
        if kind == "folder":
            if name == "Images":
                print(f"skip (images, not needed): {new_rel}")
                continue
            children = list_folder(path + "/")
            crawl(children, local_dir, new_rel)
        else:
            if not name.endswith(".nii.gz"):
                continue
            out_dir = os.path.join(local_dir, os.path.dirname(new_rel))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(local_dir, new_rel)
            size = entry["attributes"].get("size", 0)
            print(f"downloading {new_rel} ({size} bytes)")
            download_file(path, out_path)


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    root_entries = list_folder("")
    crawl(root_entries, output_dir)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1])
