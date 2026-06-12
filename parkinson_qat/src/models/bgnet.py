import torch
import torch.nn as nn
from torch import Tensor


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class SNPEncoder(nn.Module):
    """MLP leve para codificar vetor de SNPs em embedding de dim fixa."""
    def __init__(self, n_snps: int = 5, embed_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_snps, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, snp: Tensor) -> Tensor:
        return self.net(snp)


class BGNet(nn.Module):
    """
    Classificador multimodal: CNN 3D (ROIs gânglios da base) + MLP (SNPs).

    Modos:
      - image_only=True:  usa apenas RM (compatível com PTQ/QAT)
      - image_only=False: fusão CNN + SNP (modelo completo)
    """
    def __init__(self, in_channels: int = 4, base_ch: int = 32,
                 dropout: float = 0.3, n_snps: int = 5,
                 snp_embed_dim: int = 32, image_only: bool = False):
        super().__init__()
        self.image_only = image_only

        # Encoder de imagem (CNN 3D)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock3D(base_ch, base_ch)
        self.layer2 = ResBlock3D(base_ch, base_ch * 2, stride=2)
        self.layer3 = ResBlock3D(base_ch * 2, base_ch * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)

        img_feat_dim = base_ch * 4

        # Encoder genético (MLP)
        if not image_only:
            self.snp_encoder = SNPEncoder(n_snps, snp_embed_dim, dropout)
            clf_in = img_feat_dim + snp_embed_dim
        else:
            self.snp_encoder = None
            clf_in = img_feat_dim

        self.classifier = nn.Linear(clf_in, 1)

    def _encode_image(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feats = self.layer3(x)
        pooled = self.pool(feats).flatten(1)
        return feats, self.dropout(pooled)

    def forward(self, x: Tensor, snp: Tensor | None = None) -> Tensor:
        feats_map, img_emb = self._encode_image(x)

        if self.image_only or snp is None:
            fused = img_emb
        else:
            snp_emb = self.snp_encoder(snp)
            fused = torch.cat([img_emb, snp_emb], dim=1)

        return self.classifier(fused).squeeze(1)

    def get_features(self, x: Tensor,
                     snp: Tensor | None = None) -> tuple[Tensor, Tensor]:
        feats_map, img_emb = self._encode_image(x)

        if self.image_only or snp is None:
            fused = img_emb
        else:
            snp_emb = self.snp_encoder(snp)
            fused = torch.cat([img_emb, snp_emb], dim=1)

        return feats_map, self.classifier(fused).squeeze(1)
