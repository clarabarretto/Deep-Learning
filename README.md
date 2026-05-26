# Longitudinal Prediction of AMD Progression Using Foundation Models

**Maria Clara Falcão Guerra Barretto · Rodrigo Barbosa de Oliveira**  
Centro de Informática — Universidade Federal de Pernambuco

---

## Visão Geral

Pipeline modular de IA para predição longitudinal da progressão de Degeneração Macular Relacionada à Idade (AMD) a partir de imagens de fundo de olho. O projeto estende o benchmark de Cohen et al. (2025) deslocando o foco da classificação por imagem isolada para análise temporal por paciente.

**Estágio atual:** replicação do baseline (classificador binário AMD vs. non-AMD).  
**Próximo estágio:** modelagem longitudinal com comparação de embeddings entre visitas.

---

## Dataset

**HYAMD** — 1.570 imagens de fundo de olho digitais (DFIs) de 433 pacientes, adquiridas entre 2021 e 2024 no Hillel Yaffe Medical Center (câmera Topcon, 45°, resolução 1960×1934).

| AMD no CSV | Significado | Label |
|---|---|---|
| 0 | Controle / DR | 0 (non-AMD) |
| 1 | Early AMD | 0 (non-AMD) |
| 2 | Intermediário-tardio (NVAMD, GA, drusas grandes) | 1 (AMD) |

Prevalência após binarização: **27,2%** (consistente com o artigo de referência).

---

## Resultados do Baseline

| Modelo | AUROC (bootstrap, 1.000 amostras, 80% teste) |
|---|---|
| iBOT ViT-L/16 (tentativa direta) | 0.52–0.61 (instável) |
| **ViT-S/16 DINO** (baseline adaptado) | **0.8888** |
| iBOT AREDS→HYAMD OOD (artigo ref.) | 0.806 ± 0.014 |

> O ViT-S/16 DINO foi treinado dentro da mesma distribuição do HYAMD (avaliação in-distribution), o que naturalmente eleva o AUROC em relação ao cenário OOD do artigo de referência.

---

## Estrutura do Projeto

```
Deep-Learning/
├── src/
│   ├── config.py         # Todos os hiperparâmetros e caminhos em um lugar
│   ├── preprocessing.py  # crop + pad + resize para imagens de fundo de olho
│   ├── dataset.py        # FundusDataset + divisão por patient_id
│   ├── models.py         # Factory para ViT-S/16 DINO, MobileNetV3 e outros
│   ├── cam.py            # AttentionCAM_ViT + GradCAM_CNN + overlay_attention
│   ├── training.py       # Loop de treino, scheduler warmup+coseno, val loss
│   └── evaluation.py     # Bootstrap AUROC, curvas ROC, análise por subgrupo
├── outputs/              # Figuras, CSVs e checkpoints gerados (não versionado)
├── HYAMD_AMD_Detection.ipynb  # Notebook principal (orquestração no Colab)
└── README.md
```

---

## Como Usar no Google Colab

```python
# 1. Clonar o repositório
!git clone https://github.com/clarabarretto/Deep-Learning.git
import sys
sys.path.insert(0, '/content/Deep-Learning')

# 2. Montar o Drive e ajustar os caminhos em src/config.py
from google.colab import drive
drive.mount('/content/drive')

# 3. Importar os módulos
from src.config import *
from src.dataset import load_and_binarize, make_splits, FundusDataset
from src.models import build_model
from src.training import train_one_epoch, compute_val_loss, get_scheduler_with_warmup
from src.evaluation import compute_auroc, bootstrap_auroc
from src.cam import AttentionCAM_ViT, GradCAM_CNN, overlay_attention
```

---

## Módulos

| Módulo | Responsabilidade |
|---|---|
| `config.py` | Ponto único de configuração: `ARCH`, `LR`, `IMG_SIZE`, paths, seeds |
| `preprocessing.py` | `preprocess_fundus` — crop → pad → resize (Seção 3 do artigo ref.) |
| `dataset.py` | `load_and_binarize`, `make_splits` (sem leakage por patient_id), `FundusDataset` |
| `models.py` | `build_model(arch)` com registro de arquiteturas; racional de escala no código |
| `cam.py` | `AttentionCAM_ViT` (CLS-to-patch, média entre cabeças), `GradCAM_CNN`, `overlay_attention` |
| `training.py` | Warmup linear + coseno, gradient clipping, loop com early stopping |
| `evaluation.py` | Bootstrap AUROC, curva ROC, AUROC por faixa etária, `save_summary` |

---

## Referências

- Cohen et al. (2025). *Benchmarking Ophthalmology Foundation Models for Clinically Significant Age Macular Degeneration Detection.* arXiv:2505.05291
- Meisel et al. (2025). *HYAMD high-resolution fundus image dataset for AMD diagnosis.* PhysioNet
- Caron et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers (DINO).* ICCV
- Selvaraju et al. (2020). *Grad-CAM.* IJCV
