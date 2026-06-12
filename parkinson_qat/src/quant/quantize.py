import copy
import json
import torch
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qconfig,
    get_default_qat_qconfig,
    prepare,
    prepare_qat,
    convert,
    fuse_modules,
)
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def fuse_model(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model)
    fuse_modules(model, [["stem.0", "stem.1", "stem.2"]], inplace=True)
    for layer in [model.layer1, model.layer2, model.layer3]:
        fuse_modules(layer, [["conv1", "bn1"], ["conv2", "bn2"]], inplace=True)
    return model


def apply_ptq(model: nn.Module, calibration_loader,
              device: torch.device, backend: str = "fbgemm") -> nn.Module:
    model_fused = fuse_model(model)
    model_fused.eval().cpu()
    model_fused.qconfig = get_default_qconfig(backend)
    prepare(model_fused, inplace=True)

    with torch.no_grad():
        for x, _, _ in calibration_loader:
            model_fused(x.cpu())

    convert(model_fused, inplace=True)
    print("PTQ INT8 aplicado.")
    return model_fused


def apply_qat(model: nn.Module, train_loader, val_loader,
              device: torch.device, config: dict,
              out_dir: Path, backend: str = "fbgemm") -> nn.Module:
    model_fused = fuse_model(model)
    model_fused.train()
    model_fused.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(model_fused, inplace=True)
    model_fused = model_fused.to(device)

    qat_lr = config["lr"] / 10
    optimizer = Adam(model_fused.parameters(), lr=qat_lr, weight_decay=config["wd"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["qat_epochs"])
    pos_weight = torch.tensor([config["pos_weight"]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0.0
    history = []

    for epoch in range(1, config["qat_epochs"] + 1):
        model_fused.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model_fused(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model_fused.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                logits = model_fused(x)
                preds.extend(torch.sigmoid(logits).cpu().tolist())
                targets.extend(y.tolist())

        val_auc = roc_auc_score(targets, preds) if len(set(targets)) > 1 else 0.0
        history.append({"epoch": epoch, "val_auc": val_auc})

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model_fused.state_dict(), out_dir / "best_qat.pt")

        if epoch % 5 == 0:
            print(f"  QAT Epoch {epoch:3d} | val_auc={val_auc:.4f}")

    model_fused.cpu().eval()
    convert(model_fused, inplace=True)

    with open(out_dir / "qat_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"QAT INT8 concluído. Best AUC: {best_auc:.4f}")
    return model_fused


def measure_model(model: nn.Module, input_shape: tuple,
                  n_runs: int = 50) -> dict:
    import time
    dummy = torch.zeros(1, *input_shape)
    model.eval().cpu()

    with torch.no_grad():
        for _ in range(5):
            model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    size_mb = sum(p.numel() * p.element_size()
                  for p in model.parameters()) / (1024 ** 2)

    return {
        "latency_mean_ms": round(sum(times) / len(times), 3),
        "latency_std_ms": round((sum((t - sum(times)/len(times))**2
                                     for t in times) / len(times)) ** 0.5, 3),
        "size_mb": round(size_mb, 3),
    }
