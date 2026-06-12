import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json


class GradCAM3D:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._hooks.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor) -> np.ndarray:
        self.model.eval()
        x = x.unsqueeze(0).requires_grad_(True)
        logits = self.model(x)
        self.model.zero_grad()
        logits.backward()

        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def bg_concentration(cam: np.ndarray, threshold: float = 0.5) -> float:
    return float((cam > threshold).mean())


def spectral_analysis(model: torch.nn.Module,
                      loader, device: torch.device,
                      max_batches: int = 20) -> dict:
    model.eval()
    all_features = []

    with torch.no_grad():
        for i, (x, _, _) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            feats, _ = model.get_features(x)
            feats_flat = feats.flatten(2).mean(dim=2)
            all_features.append(feats_flat.cpu())

    F_mat = torch.cat(all_features, dim=0).numpy()
    F_centered = F_mat - F_mat.mean(axis=0)

    _, s, _ = np.linalg.svd(F_centered, full_matrices=False)
    s_norm = s / s.sum()
    cumvar = np.cumsum(s_norm)

    effective_rank = int(np.searchsorted(cumvar, 0.90)) + 1

    return {
        "singular_values": s.tolist(),
        "singular_values_norm": s_norm.tolist(),
        "cumulative_variance": cumvar.tolist(),
        "effective_rank_90": effective_rank,
        "n_components": len(s),
    }


def run_analysis(models_dict: dict, val_loader,
                 device: torch.device, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, model in models_dict.items():
        print(f"\nAnalisando: {name}")
        model = model.to(device)

        spec = spectral_analysis(model, val_loader, device)
        print(f"  Rank efetivo (90% var): {spec['effective_rank_90']}")

        try:
            target_layer = model.layer3
            gcam = GradCAM3D(model, target_layer)
            concentrations = []
            for x, y, _ in val_loader:
                for i in range(min(len(x), 5)):
                    cam = gcam.generate(x[i].to(device))
                    concentrations.append(bg_concentration(cam))
            gcam.remove_hooks()
            mean_conc = float(np.mean(concentrations))
            print(f"  Concentração Grad-CAM na ROI: {mean_conc:.4f}")
        except Exception as e:
            mean_conc = None
            print(f"  Grad-CAM falhou: {e}")

        results[name] = {
            "spectral": spec,
            "gradcam_bg_concentration": mean_conc,
        }

    with open(out_dir / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResultados salvos em {out_dir / 'analysis_results.json'}")
    return results
