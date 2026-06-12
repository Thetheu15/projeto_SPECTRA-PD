# Parkinson QAT — Basal Ganglia Classifier

Estudo do efeito de Quantization-Aware Training (QAT) como regularizador espectral
em modelos de detecção de Parkinson via RM T1 dos gânglios da base (PPMI dataset).

## Estrutura

```
parkinson_qat/
├── configs/            # Hiperparâmetros
├── data/               # Dados (não versionados)
│   ├── raw/            # NIfTI baixados do LONI IDA
│   ├── processed/      # Pós ANTs (skull strip + MNI)
│   └── splits/         # ROIs extraídas + splits.json
├── src/
│   ├── data/           # Dataset PyTorch
│   ├── models/         # BGNet (ResNet3D)
│   ├── quant/          # PTQ e QAT
│   └── analysis/       # Grad-CAM e SVD espectral
├── scripts/            # Pré-processamento e extração de ROI
├── experiments/        # Script principal do experimento
└── outputs/            # Checkpoints, figuras, resultados
```

## Reprodução

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Pré-processar imagens (ANTs: skull strip + registro MNI)
python scripts/01_preprocess.py \
  --pd_csv data/ppmi_pd_final_323.csv \
  --hc_csv data/ppmi_hc_final.csv \
  --raw_dir data/raw \
  --out_dir data/processed

# 3. Extrair ROIs dos gânglios da base (atlas MNI)
python scripts/02_extract_roi.py \
  --processed_dir data/processed \
  --atlas_path data/atlas/CIT168_basal_ganglia_1mm.nii.gz \
  --out_dir data/splits

# 4. Gerar splits estratificados (5-fold CV)
python -c "
from src.data.dataset import make_splits
make_splits('data/splits/manifest.json', n_folds=5)
"

# 5. Rodar experimento completo (FP32 + PTQ + QAT + análise)
python experiments/run_experiment.py \
  --config configs/base.json \
  --splits data/splits/splits.json \
  --fold 0 \
  --out_dir outputs/experiment_fold0
```

## Dataset

- **PD:** 323 sujeitos | **HC:** 108 sujeitos | **Total:** 431
- Fonte: PPMI (ppmi-info.org), Application ID 124382
- Imagens: T1 MPRAGE 3T Siemens, baseline
- Ratio PD:HC = 3:1 → tratado com `pos_weight=3.0` na BCEWithLogitsLoss

## Hipótese central

QAT age como regularizador espectral implícito, suprimindo componentes de alta
variância sensíveis ao arredondamento e forçando o modelo a depender de features
diagnósticas de baixa variância nos gânglios da base — medido via rank efetivo
(SVD) e concentração de Grad-CAM nas ROIs segmentadas.

## Citação

```
Monteiro, A.M. (2026). QAT as Spectral Regularizer for Basal Ganglia Features
in Parkinson's Disease Detection. [Conference paper]
```
