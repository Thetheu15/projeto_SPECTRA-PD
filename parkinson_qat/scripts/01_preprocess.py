import os
import argparse
import json
import ants
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


MNI_TEMPLATE = "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"


def preprocess_subject(input_path: Path, output_path: Path, mni_template: str) -> dict:
    img = ants.image_read(str(input_path))

    img_n4 = ants.n4_bias_field_correction(img)

    brain_mask = ants.get_mask(img_n4)
    img_brain = ants.mask_image(img_n4, brain_mask)

    template = ants.image_read(mni_template)
    registration = ants.registration(
        fixed=template,
        moving=img_brain,
        type_of_transform="Affine",
    )
    img_mni = registration["warpedmovout"]

    arr = img_mni.numpy()
    p1, p99 = np.percentile(arr[arr > 0], [1, 99])
    arr_clipped = np.clip(arr, p1, p99)
    arr_norm = (arr_clipped - arr_clipped.mean()) / (arr_clipped.std() + 1e-8)
    img_final = ants.from_numpy(arr_norm, origin=img_mni.origin,
                                spacing=img_mni.spacing, direction=img_mni.direction)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(img_final, str(output_path))

    return {
        "input": str(input_path),
        "output": str(output_path),
        "shape": list(arr_norm.shape),
        "spacing": list(img_mni.spacing),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pd_csv", required=True, help="CSV com sujeitos PD (Subject ID, Description)")
    parser.add_argument("--hc_csv", required=True, help="CSV com sujeitos HC")
    parser.add_argument("--raw_dir", required=True, help="Diretório raiz com imagens baixadas do LONI")
    parser.add_argument("--out_dir", required=True, help="Diretório de saída para imagens processadas")
    parser.add_argument("--mni", default=MNI_TEMPLATE, help="Template MNI T1 1mm")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd_df = pd.read_csv(args.pd_csv)
    pd_df["label"] = 1
    hc_df = pd.read_csv(args.hc_csv)
    hc_df["label"] = 0
    all_subjects = pd.concat([pd_df, hc_df], ignore_index=True)

    log = []
    failed = []

    for _, row in tqdm(all_subjects.iterrows(), total=len(all_subjects)):
        sid = str(row["Subject ID"])
        label = row["label"]
        group = "PD" if label == 1 else "HC"

        candidates = list(raw_dir.rglob(f"*{sid}*.nii*"))
        if not candidates:
            failed.append({"subject": sid, "reason": "file not found"})
            continue

        input_path = candidates[0]
        output_path = out_dir / group / f"{sid}.nii.gz"

        try:
            info = preprocess_subject(input_path, output_path, args.mni)
            info["subject_id"] = sid
            info["label"] = label
            log.append(info)
        except Exception as e:
            failed.append({"subject": sid, "reason": str(e)})

    with open(out_dir / "preprocess_log.json", "w") as f:
        json.dump({"processed": log, "failed": failed}, f, indent=2)

    print(f"\nProcessados: {len(log)}")
    print(f"Falhos: {len(failed)}")
    if failed:
        print("Falhos:", [f["subject"] for f in failed])


if __name__ == "__main__":
    main()
