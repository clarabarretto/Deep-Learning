from .config import (
    HYAMD_ROOT, IMAGES_DIR, LABELS_CSV, OUTPUT_DIR,
    ARCH, ARCH_LR, LR,
    SEED, BATCH_SIZE, NUM_EPOCHS, WEIGHT_DECAY, WARMUP_EPOCHS, PATIENCE,
    IMG_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD,
    BOOTSTRAP_N, BOOTSTRAP_FRAC, OVERFIT_RATIO_MAX,
)
from .preprocessing import preprocess_fundus, crop_fundus_background, pad_to_square
from .dataset import FundusDataset, load_and_binarize, make_splits
from .models import build_model, ARCH_REGISTRY
from .cam import AttentionCAM_ViT, GradCAM_CNN, overlay_attention
from .training import get_scheduler_with_warmup, train_one_epoch, compute_val_loss
from .evaluation import (
    compute_auroc, bootstrap_auroc,
    plot_training_curves, plot_roc_and_probs, plot_auroc_by_age,
    save_summary,
)

__all__ = [
    # config
    'HYAMD_ROOT', 'IMAGES_DIR', 'LABELS_CSV', 'OUTPUT_DIR',
    'ARCH', 'ARCH_LR', 'LR',
    'SEED', 'BATCH_SIZE', 'NUM_EPOCHS', 'WEIGHT_DECAY', 'WARMUP_EPOCHS', 'PATIENCE',
    'IMG_SIZE', 'NUM_WORKERS',
    'IMAGENET_MEAN', 'IMAGENET_STD',
    'BOOTSTRAP_N', 'BOOTSTRAP_FRAC', 'OVERFIT_RATIO_MAX',
    # preprocessing
    'preprocess_fundus', 'crop_fundus_background', 'pad_to_square',
    # dataset
    'FundusDataset', 'load_and_binarize', 'make_splits',
    # models
    'build_model', 'ARCH_REGISTRY',
    # cam
    'AttentionCAM_ViT', 'GradCAM_CNN', 'overlay_attention',
    # training
    'get_scheduler_with_warmup', 'train_one_epoch', 'compute_val_loss',
    # evaluation
    'compute_auroc', 'bootstrap_auroc',
    'plot_training_curves', 'plot_roc_and_probs', 'plot_auroc_by_age',
    'save_summary',
]
