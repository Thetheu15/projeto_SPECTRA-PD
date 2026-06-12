import argparse
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm


ATLAS_URL = "https://www.nitrc.org/frs/download.php/11669/CIT168toMNI152_prob_atlas_bilat_1mm.nii.gz"

BG_LABELS = {
    "putamen": [3, 4],
    "caudate": [1, 2],
    "pallidum": [5, 6],
    "thalamus": [9, 10],
}

ROI_CROP_SIZE = (32, 32, 32)


def load_atlas(atlas_path: str) -> np.ndarray:
    img = nib.load(atlas_path)
    return img.get_fdata()


def extract_roi(volume: np.ndarray, atlas: np.ndarray,
                labels: list, crop_size: tuple) -> np.ndarray:
    mask = np.isin(atlas, labels).astype(np.float32)
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return np.zeros(crop_size, dtype=np.float32)

    cx = int(np.mean(coords[0]))
    cy = int(np.mean(coords[1]))
    cz = int(np.mean(coords[2]))

    hx, hy, hz = [s // 2 for s in crop_size]

    x0, x1 = max(0, cx - hx), min(volume.shape[0], cx + hx)
    y0, y1 = max(0, cy - hy), min(volume.shape[1], cy + hy)
    z0, z1 = max(0, cz - hz), min(volume.shape[2], cz + hz)

    roi = volume[x0:x1, y0:y1, z0:z1].astype(np.float32)

    pad = [(0, max(0, crop_size[i] - roi.shape[i])) for i in range(3)]
    roi = np.pad(roi, pad, mode="constant")
    roi = roi[:crop_size[0], :crop_size[1], :crop_size[2]]

    return roi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--atlas_path", required=True, help="Atlas MNI em espaço 1mm")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--crop_size", default=32, type=int)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop = (args.crop_size,) * 3

    atlas = load_atlas(args.atlas_path)

    records = []
    for group, label in [("PD", 1), ("HC", 0)]:
        group_dir = processed_dir / group
        if not group_dir.exists():
            continue
        for nii_path in tqdm(sorted(group_dir.glob("*.nii.gz")), desc=group):
            sid = nii_path.stem.replace(".nii", "")
            vol = nib.load(str(nii_path)).get_fdata().astype(np.float32)

            rois = {}
            for region, lbls in BG_LABELS.items():
                rois[region] = extract_roi(vol, atlas, lbls, crop)

            bg_volume = np.stack(list(rois.values()), axis=0)

            out_path = out_dir / group / f"{sid}.npy"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), bg_volume)

            records.append({
                "subject_id": sid,
                "label": label,
                "path": str(out_path),
                "shape": list(bg_volume.shape),
            })

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(records, f, indent=2)

    pd_n = sum(1 for r in records if r["label"] == 1)
    hc_n = sum(1 for r in records if r["label"] == 0)
    print(f"\nROIs extraídas: {len(records)} ({pd_n} PD, {hc_n} HC)")
    print(f"Shape por sujeito: {records[0]['shape'] if records else 'N/A'}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
