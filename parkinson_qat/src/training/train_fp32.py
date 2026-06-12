import json
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from pathlib import Path

from src.models.bgnet import BGNet
from src.data.dataset import get_dataloaders, load_snp_features


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, preds, targets = 0.0, [], []
    for x, snp, y, _ in loader:
        x, snp, y = x.to(device), snp.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, snp)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds.extend(torch.sigmoid(logits).detach().cpu().tolist())
        targets.extend(y.cpu().tolist())
    auc = roc_auc_score(targets, preds) if len(set(targets)) > 1 else 0.0
    return total_loss / len(loader), auc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for x, snp, y, _ in loader:
        x, snp, y = x.to(device), snp.to(device), y.to(device)
        logits = model(x, snp)
        loss = criterion(logits, y)
        total_loss += loss.item()
        preds.extend(torch.sigmoid(logits).cpu().tolist())
        targets.extend(y.cpu().tolist())
    auc = roc_auc_score(targets, preds) if len(set(targets)) > 1 else 0.0
    return total_loss / len(loader), auc


def train(config: dict, fold: dict, out_dir: Path,
          device: torch.device, snp_features: dict | None = None):
    train_dl, val_dl = get_dataloaders(
        fold, snp_features=snp_features,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    image_only = snp_features is None
    model = BGNet(
        in_channels=config["in_channels"],
        base_ch=config["base_ch"],
        dropout=config["dropout"],
        n_snps=config.get("n_snps", 5),
        snp_embed_dim=config.get("snp_embed_dim", 32),
        image_only=image_only,
    ).to(device)

    pos_weight = torch.tensor([config["pos_weight"]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_auc, history = 0.0, []

    for epoch in range(1, config["epochs"] + 1):
        tr_loss, tr_auc = train_epoch(model, train_dl, optimizer, criterion, device)
        val_loss, val_auc = eval_epoch(model, val_dl, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "tr_loss": tr_loss, "tr_auc": tr_auc,
            "val_loss": val_loss, "val_auc": val_auc,
        })

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | tr_auc={tr_auc:.3f} val_auc={val_auc:.3f}")

    torch.save(model.state_dict(), out_dir / "last_model.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Best val AUC: {best_auc:.4f}")
    return model, best_auc, train_dl, val_dl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--snp_features", default=None)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--out_dir", default="outputs/fp32")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    with open(args.splits) as f:
        folds = json.load(f)

    snp_features = load_snp_features(args.snp_features) if args.snp_features else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(config, folds[args.fold], out_dir, device, snp_features)


if __name__ == "__main__":
    main()
