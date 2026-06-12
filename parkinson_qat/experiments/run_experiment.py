import json
import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from src.models.bgnet import BGNet
from src.data.dataset import get_dataloaders, load_snp_features
from src.training.train_fp32 import train
from src.quant.quantize import apply_ptq, apply_qat, measure_model
from src.analysis.spectral_gradcam import run_analysis


@torch.no_grad()
def evaluate(model, loader, device, use_snp: bool = True) -> dict:
    model.eval().cpu()
    preds_prob, preds_bin, targets = [], [], []
    for x, snp, y, _ in loader:
        snp_in = snp if use_snp else None
        logits = model(x, snp_in)
        prob = torch.sigmoid(logits).tolist()
        preds_prob.extend(prob)
        preds_bin.extend([1 if p > 0.5 else 0 for p in prob])
        targets.extend(y.tolist())
    return {
        "auc": round(roc_auc_score(targets, preds_prob), 4),
        "acc": round(accuracy_score(targets, preds_bin), 4),
        "f1":  round(f1_score(targets, preds_bin), 4),
    }


def run_fold(config, fold, fold_idx, snp_features, out_dir, device):
    fold_dir = out_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Treinar FP32 image-only (baseline sem SNP) ──────────────────────
    print(f"\n  [FP32-IMG] Treinando sem SNP...")
    img_dir = fold_dir / "fp32_img"
    img_dir.mkdir(parents=True, exist_ok=True)
    model_img, _, train_dl, val_dl = train(
        config, fold, img_dir, device, snp_features=None
    )
    model_img.load_state_dict(torch.load(img_dir / "best_model.pt"))

    # ── 2. Treinar FP32 multimodal (imagem + SNP) ──────────────────────────
    print(f"\n  [FP32-MM] Treinando com SNP...")
    mm_dir = fold_dir / "fp32_multimodal"
    mm_dir.mkdir(parents=True, exist_ok=True)
    model_mm, _, train_dl_snp, val_dl_snp = train(
        config, fold, mm_dir, device, snp_features=snp_features
    )
    model_mm.load_state_dict(torch.load(mm_dir / "best_model.pt"))

    # ── 3. PTQ no modelo multimodal ────────────────────────────────────────
    print(f"\n  [PTQ] Quantizando modelo multimodal...")
    ptq_dir = fold_dir / "ptq_multimodal"
    ptq_dir.mkdir(parents=True, exist_ok=True)

    # PTQ opera no modo image_only para compatibilidade com torch.ao.quantization
    model_ptq_base = BGNet(
        in_channels=config["in_channels"], base_ch=config["base_ch"],
        dropout=0.0, image_only=True
    )
    img_state = {k: v for k, v in
                 torch.load(mm_dir / "best_model.pt").items()
                 if not k.startswith("snp_encoder") and k != "classifier.weight"
                 and k != "classifier.bias"}
    model_ptq_base.load_state_dict(img_state, strict=False)
    model_ptq = apply_ptq(model_ptq_base, val_dl, device)
    torch.save(model_ptq.state_dict(), ptq_dir / "model_ptq.pt")

    # ── 4. QAT no modelo multimodal ────────────────────────────────────────
    print(f"\n  [QAT] QAT no modelo multimodal...")
    qat_dir = fold_dir / "qat_multimodal"
    qat_dir.mkdir(parents=True, exist_ok=True)

    model_qat_base = BGNet(
        in_channels=config["in_channels"], base_ch=config["base_ch"],
        dropout=config["dropout"], image_only=True
    )
    model_qat_base.load_state_dict(img_state, strict=False)
    model_qat = apply_qat(
        model_qat_base, train_dl, val_dl, device, config, qat_dir
    )

    # ── 5. Avaliação comparativa ───────────────────────────────────────────
    print(f"\n  [EVAL] Avaliando todos os modelos...")
    input_shape = (config["in_channels"], 32, 32, 32)

    results = {}

    # FP32 image-only
    m = evaluate(model_img, val_dl, device, use_snp=False)
    p = measure_model(model_img, input_shape)
    results["FP32_image_only"] = {**m, **p, "modality": "image"}

    # FP32 multimodal
    m = evaluate(model_mm, val_dl_snp, device, use_snp=True)
    p = measure_model(model_mm, input_shape)
    results["FP32_multimodal"] = {**m, **p, "modality": "image+snp"}

    # PTQ image-only (para comparação de quantização)
    m = evaluate(model_ptq, val_dl, device, use_snp=False)
    p = measure_model(model_ptq, input_shape)
    results["PTQ_INT8"] = {**m, **p, "modality": "image"}

    # QAT image-only
    m = evaluate(model_qat, val_dl, device, use_snp=False)
    p = measure_model(model_qat, input_shape)
    results["QAT_INT8"] = {**m, **p, "modality": "image"}

    for name, r in results.items():
        print(f"    {name:25s} AUC={r['auc']} ACC={r['acc']} F1={r['f1']} "
              f"| {r['size_mb']}MB {r['latency_mean_ms']}ms")

    with open(fold_dir / "ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── 6. Análise espectral + Grad-CAM ───────────────────────────────────
    print(f"\n  [ANALYSIS] Análise espectral e Grad-CAM...")
    analysis_models = {
        "FP32_img": model_img.to(device),
        "FP32_mm":  model_mm.to(device),
    }
    run_analysis(analysis_models, val_dl_snp, device, fold_dir / "analysis")

    return results


def aggregate_folds(all_fold_results: list, out_dir: Path):
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))

    for fold_res in all_fold_results:
        for model_name, metrics in fold_res.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    agg[model_name][k].append(v)

    summary = {}
    for model_name, metric_lists in agg.items():
        summary[model_name] = {
            k: {
                "mean": round(float(np.mean(v)), 4),
                "std":  round(float(np.std(v)), 4),
            }
            for k, v in metric_lists.items()
        }

    with open(out_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CROSS-VALIDATION SUMMARY ===")
    for model_name, metrics in summary.items():
        auc = metrics.get("auc", {})
        print(f"  {model_name:25s} AUC={auc.get('mean','?')} ± {auc.get('std','?')}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--snp_features", default=None,
                        help="JSON gerado por scripts/03_process_snps.py")
    parser.add_argument("--folds", default="0,1,2,3,4",
                        help="Folds a rodar, ex: '0,1,2,3,4'")
    parser.add_argument("--out_dir", default="outputs/experiment")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    with open(args.splits) as f:
        all_folds = json.load(f)

    snp_features = load_snp_features(args.snp_features) if args.snp_features else None
    fold_ids = [int(i) for i in args.folds.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Folds: {fold_ids}")
    print(f"SNP features: {'sim' if snp_features else 'não (mode image-only)'}\n")

    all_results = []
    for fold_idx in fold_ids:
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*50}")
        res = run_fold(config, all_folds[fold_idx], fold_idx,
                       snp_features, out_dir, device)
        all_results.append(res)

    aggregate_folds(all_results, out_dir)
    print(f"\nExperimento completo. Resultados em: {out_dir}")


if __name__ == "__main__":
    main()
