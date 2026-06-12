import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


RISK_SNPS = {
    "SNCA_rs356181":  {"gene": "SNCA",  "chr": 4},
    "SNCA_rs2736990": {"gene": "SNCA",  "chr": 4},
    "LRRK2_rs34637584": {"gene": "LRRK2", "chr": 12},
    "GBA_rs76763715": {"gene": "GBA",   "chr": 1},
    "VPS52_rs213202": {"gene": "VPS52", "chr": 6},
}

SNP_IDS = list(RISK_SNPS.keys())


def load_plink_raw(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, sep=r"\s+")
    df = df.rename(columns={"IID": "subject_id"})
    df["subject_id"] = df["subject_id"].astype(str)
    return df


def load_vcf_snps(vcf_path: str, snp_ids: list) -> pd.DataFrame:
    rows = []
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            rsid = parts[2]
            if rsid in snp_ids:
                rows.append(parts)
    if not rows:
        return pd.DataFrame()
    cols = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    df = pd.DataFrame(rows, columns=cols + [f"sample_{i}" for i in range(len(rows[0]) - 9)])
    return df


def build_snp_matrix(raw_df: pd.DataFrame, snp_ids: list,
                     subject_ids: list) -> pd.DataFrame:
    available = [s for s in snp_ids if s in raw_df.columns]
    missing = [s for s in snp_ids if s not in raw_df.columns]

    if missing:
        print(f"SNPs não encontrados no arquivo (serão imputados com 0): {missing}")
        for s in missing:
            raw_df[s] = 0

    matrix = raw_df[raw_df["subject_id"].isin(subject_ids)][
        ["subject_id"] + snp_ids
    ].copy()

    not_found = set(subject_ids) - set(matrix["subject_id"].tolist())
    if not_found:
        placeholder = pd.DataFrame({
            "subject_id": list(not_found),
            **{s: [0] * len(not_found) for s in snp_ids}
        })
        matrix = pd.concat([matrix, placeholder], ignore_index=True)

    for col in snp_ids:
        matrix[col] = matrix[col].fillna(0).clip(0, 2).astype(np.float32)
        col_mean = matrix[col].mean()
        col_std = matrix[col].std() + 1e-8
        matrix[f"{col}_norm"] = (matrix[col] - col_mean) / col_std

    return matrix.set_index("subject_id")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plink_raw", required=True,
                        help=".raw gerado por: plink2 --export A --out output")
    parser.add_argument("--pd_csv", required=True)
    parser.add_argument("--hc_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd_df = pd.read_csv(args.pd_csv)
    hc_df = pd.read_csv(args.hc_csv)
    all_sids = (
        pd_df["Subject ID"].astype(str).tolist() +
        hc_df["Subject ID"].astype(str).tolist()
    )

    print(f"Carregando PLINK .raw: {args.plink_raw}")
    raw_df = load_plink_raw(args.plink_raw)
    print(f"  Sujeitos no arquivo genético: {len(raw_df)}")

    matrix = build_snp_matrix(raw_df, SNP_IDS, all_sids)

    norm_cols = [f"{s}_norm" for s in SNP_IDS]
    snp_out = {}
    for sid in all_sids:
        if sid in matrix.index:
            snp_out[sid] = matrix.loc[sid, norm_cols].tolist()
        else:
            snp_out[sid] = [0.0] * len(SNP_IDS)

    out_path = out_dir / "snp_features.json"
    with open(out_path, "w") as f:
        json.dump({
            "snp_ids": SNP_IDS,
            "n_features": len(SNP_IDS),
            "subjects": snp_out,
            "risk_snps_info": RISK_SNPS,
        }, f, indent=2)

    coverage = sum(1 for v in snp_out.values() if any(x != 0 for x in v))
    print(f"\nSNP features salvas: {out_path}")
    print(f"Sujeitos com dados genéticos reais: {coverage}/{len(all_sids)}")
    print(f"SNPs incluídos: {SNP_IDS}")


if __name__ == "__main__":
    main()
