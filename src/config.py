"""
Configuração centralizada do pipeline de detecção de AMD.

Altere ARCH e as variáveis de PATH para trocar de arquitetura ou localização
do dataset sem precisar editar o notebook.
"""

# ── Caminhos ───────────────────────────────────────────────────────────────

HYAMD_ROOT = r'C:\Deep-Learning\hyamd-high-resolution-fundus-image-dataset-for-age-related-macular-degeneration-amd-diagnosis-1.0.0'
IMAGES_DIR = fr'{HYAMD_ROOT}\Images'
LABELS_CSV = fr'{HYAMD_ROOT}\labels\labels.csv'
OUTPUT_DIR = r'C:\Deep-Learning\outputs'

# ── Seleção de arquitetura ─────────────────────────────────────────────────
# Opções: 'ibot', 'mobilenetv3', 'efficientnet_lite0', 'convnext_femto'

ARCH = 'mobilenetv3'

# LR ajustado por arquitetura (Seção III-E/F do relatório)
ARCH_LR = {
    'ibot'              : 1e-4,
    'mobilenetv3'       : 5e-4,
    'efficientnet_lite0': 2e-4,
    'convnext_femto'    : 2e-4,
}

LR = ARCH_LR[ARCH]

# ── Hiperparâmetros de treinamento ─────────────────────────────────────────

SEED          = 42
BATCH_SIZE    = 16           # reduzido para rodar em CPU sem estourar memória
NUM_EPOCHS    = 25
WEIGHT_DECAY  = 0.01
WARMUP_EPOCHS = 3
PATIENCE      = 5
IMG_SIZE      = 224
NUM_WORKERS   = 0            # 0 é obrigatório no Windows com multiprocessing do DataLoader

# ── Normalização ImageNet ──────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Avaliação (protocolo bootstrap do artigo de referência) ────────────────

BOOTSTRAP_N           = 1000
BOOTSTRAP_FRAC        = 0.80
OVERFIT_RATIO_MAX     = 10.0
