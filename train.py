"""
Script principal para treinar e avaliar o modelo de detecção de AMD no HYAMD.

Uso:
    python train.py                          # usa ARCH definido em config.py
    python train.py --arch mobilenetv3
    python train.py --arch efficientnet_lite0
    python train.py --arch convnext_femto
    python train.py --arch ibot

Outputs salvos em config.OUTPUT_DIR:
    best_<arch>_hyamd.pth       modelo com melhor AUROC de validação
    training_curves.png
    roc_curve_and_probs.png
    auroc_by_age.png
    cam_<arch>.png              mapas de explicabilidade (Grad-CAM ou AttentionCAM)
    test_predictions.csv
    results_summary.json
    training_history.csv
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

from src import config as cfg
from src.cam import AttentionCAM_ViT, GradCAM_CNN, overlay_attention
from src.dataset import FundusDataset, load_and_binarize, make_splits
from src.evaluation import (
    bootstrap_auroc,
    compute_auroc,
    plot_auroc_by_age,
    plot_roc_and_probs,
    plot_training_curves,
    save_summary,
)
from src.models import build_model
from src.preprocessing import preprocess_fundus
from src.training import compute_val_loss, get_scheduler_with_warmup, train_one_epoch


def generate_cam_plots(model: nn.Module, results_df: pd.DataFrame,
                       arch: str, val_tf, device: torch.device,
                       output_dir: str) -> None:
    """Gera e salva mapas CAM para 1 amostra non-AMD e 1 AMD do conjunto de teste."""
    print('Gerando mapas de explicabilidade (CAM)...')

    non_amd = results_df[results_df['label'] == 0].sample(1, random_state=cfg.SEED)
    amd     = results_df[results_df['label'] == 1].sample(1, random_state=cfg.SEED)
    samples = pd.concat([non_amd, amd]).reset_index(drop=True)

    cam_gen = (
        AttentionCAM_ViT(model)
        if arch == 'ibot'
        else GradCAM_CNN(model, arch=arch)
    )

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    label_names = {0: 'non-AMD (label=0)', 1: 'AMD moderado-tardio (label=1)'}

    for col, (_, row) in enumerate(samples.iterrows()):
        img_pil    = preprocess_fundus(Image.open(row['path']).convert('RGB'))
        img_tensor = val_tf(img_pil).unsqueeze(0).to(device)

        cam = cam_gen.generate(img_tensor)

        with torch.no_grad():
            prob = torch.sigmoid(model(img_tensor)).item()

        overlay = overlay_attention(img_pil, cam)

        axes[0][col].imshow(img_pil)
        axes[0][col].set_title(
            f"{label_names[row['label']]}\np={prob:.2f} | pred={'AMD' if prob>=0.5 else 'non-AMD'}",
            fontsize=9,
        )
        axes[0][col].axis('off')

        cam_label = 'Attention Map' if arch == 'ibot' else 'Grad-CAM'
        axes[1][col].imshow(overlay)
        axes[1][col].set_title(cam_label, fontsize=9)
        axes[1][col].axis('off')

    if arch == 'ibot':
        cam_gen.remove_hook()
    else:
        cam_gen.remove_hooks()

    plt.suptitle(f'Mapas de Explicabilidade — {arch}', fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'cam_{arch}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Salvo em: {out_path}')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms(img_size: int, mean: list, std: list):
    train_tf = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


def main(arch: str) -> None:
    set_seed(cfg.SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo : {device}')
    print(f'Arquitetura : {arch}')
    print()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ── Dados ──────────────────────────────────────────────────────────────
    print('Carregando dataset...')
    df = load_and_binarize(cfg.LABELS_CSV, cfg.IMAGES_DIR)

    n_pos = int((df['label'] == 1).sum())
    n_neg = int((df['label'] == 0).sum())
    print(f'  Total : {len(df)} imagens')
    print(f'  AMD   : {n_pos}  |  non-AMD: {n_neg}  |  prevalência: {n_pos/len(df)*100:.1f}%')
    print()

    train_df, val_df, test_df = make_splits(df, seed=cfg.SEED)
    print('Split por patient_id (sem data leakage):')
    for name, sp in [('Treino', train_df), ('Validação', val_df), ('Teste', test_df)]:
        n_amd = int(sp['label'].sum())
        print(f'  {name:<10} {len(sp):>5} imgs | {sp["patient_id"].nunique():>4} pacientes '
              f'| AMD={n_amd} non-AMD={len(sp)-n_amd}')
    print()

    train_tf, val_tf = make_transforms(cfg.IMG_SIZE, cfg.IMAGENET_MEAN, cfg.IMAGENET_STD)

    train_loader = DataLoader(
        FundusDataset(train_df, transform=train_tf, target_size=cfg.IMG_SIZE),
        batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        FundusDataset(val_df, transform=val_tf, target_size=cfg.IMG_SIZE),
        batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == 'cuda'),
    )
    test_loader = DataLoader(
        FundusDataset(test_df, transform=val_tf, target_size=cfg.IMG_SIZE),
        batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == 'cuda'),
    )

    # ── Modelo ─────────────────────────────────────────────────────────────
    print('Construindo modelo...')
    model = build_model(arch, num_classes=1).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parâmetros totais    : {total/1e6:.1f}M')
    print(f'  Parâmetros treináveis: {trainable/1e6:.1f}M')
    print()

    # ── Otimizador e scheduler ─────────────────────────────────────────────
    n_neg_train = int((train_df['label'] == 0).sum())
    n_pos_train = int((train_df['label'] == 1).sum())
    pos_weight  = torch.tensor([n_neg_train / n_pos_train], dtype=torch.float32).to(device)
    print(f'pos_weight = {pos_weight.item():.3f}  (balanceia classes no treino)')

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.ARCH_LR[arch],
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    total_steps  = cfg.NUM_EPOCHS * len(train_loader)
    warmup_steps = cfg.WARMUP_EPOCHS * len(train_loader)
    scheduler    = get_scheduler_with_warmup(optimizer, warmup_steps, total_steps)
    print(f'Total steps: {total_steps} | Warmup: {warmup_steps} '
          f'({warmup_steps/total_steps*100:.1f}%)')
    print()

    # ── Loop de treinamento ────────────────────────────────────────────────
    best_model_path   = os.path.join(cfg.OUTPUT_DIR, f'best_{arch}_hyamd.pth')
    history           = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_auroc': []}
    best_val_auroc    = 0.0
    best_val_loss     = float('inf')
    epochs_no_improve = 0

    print(f'Treinando por até {cfg.NUM_EPOCHS} épocas (patience={cfg.PATIENCE})...')
    print('-' * 72)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_auroc  = compute_auroc(model, val_loader, device)
        val_loss   = compute_val_loss(model, val_loader, criterion, device)

        overfit_ratio = val_loss / max(train_loss, 1e-8)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)

        auroc_improved = val_auroc > best_val_auroc
        not_overfit    = overfit_ratio < cfg.OVERFIT_RATIO_MAX
        flag = ''

        if auroc_improved and not_overfit:
            best_val_auroc    = val_auroc
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            flag = ' <- melhor'
        else:
            epochs_no_improve += 1

        print(
            f'Época {epoch:02d}/{cfg.NUM_EPOCHS} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_loss:.4f} | '
            f'Overfit: {overfit_ratio:5.1f}x | '
            f'Val AUROC: {val_auroc:.4f}'
            f'{flag}'
        )

        if epochs_no_improve >= cfg.PATIENCE:
            print(f'\nEarly stopping na época {epoch} (sem melhora há {cfg.PATIENCE} épocas).')
            break

    print('-' * 72)
    print(f'Melhor Val AUROC: {best_val_auroc:.4f}')
    print()

    # ── Avaliação final ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            logits = model(imgs.to(device)).squeeze(1)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    bs_mean, bs_std = bootstrap_auroc(
        all_labels, all_probs,
        n_bootstrap=cfg.BOOTSTRAP_N,
        sample_frac=cfg.BOOTSTRAP_FRAC,
        seed=cfg.SEED,
    )
    print(f'AUROC Bootstrap (1000×, 80%): {bs_mean:.3f} ± {bs_std:.3f}')
    print()

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_training_curves(history, best_val_auroc, arch, cfg.OUTPUT_DIR)
    plot_roc_and_probs(all_labels, all_probs, bs_mean, bs_std, arch, cfg.OUTPUT_DIR)

    # ── Análise por subgrupo etário ────────────────────────────────────────
    results_df = test_df.copy().reset_index(drop=True)
    results_df['pred_prob']  = all_probs
    results_df['pred_label'] = (all_probs >= 0.5).astype(int)
    results_df['correct']    = (results_df['pred_label'] == results_df['label']).astype(int)
    results_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'test_predictions.csv'), index=False)

    print('AUROC por faixa etária:')
    non_amd_mask = results_df['AMD'].isin([0, 1])
    plot_auroc_by_age(results_df, non_amd_mask, cfg.OUTPUT_DIR)
    print()

    # ── Mapas de explicabilidade (CAM) ─────────────────────────────────────
    _, val_tf = make_transforms(cfg.IMG_SIZE, cfg.IMAGENET_MEAN, cfg.IMAGENET_STD)
    generate_cam_plots(model, results_df, arch, val_tf, device, cfg.OUTPUT_DIR)
    print()

    # ── Resumo ─────────────────────────────────────────────────────────────
    save_summary({
        'dataset'              : 'HYAMD (local)',
        'modelo'               : arch,
        'binarizacao'          : 'AMD=2 -> 1 | AMD in {0,1} -> 0',
        'split_por'            : 'patient_id (sem data leakage)',
        'n_train_imgs'         : int(len(train_df)),
        'n_val_imgs'           : int(len(val_df)),
        'n_test_imgs'          : int(len(test_df)),
        'auroc_bootstrap_mean' : round(float(bs_mean), 3),
        'auroc_bootstrap_std'  : round(float(bs_std),  3),
        'epochs'               : cfg.NUM_EPOCHS,
        'batch_size'           : cfg.BATCH_SIZE,
        'lr'                   : cfg.ARCH_LR[arch],
        'weight_decay'         : cfg.WEIGHT_DECAY,
        'warmup_epochs'        : cfg.WARMUP_EPOCHS,
        'img_size'             : cfg.IMG_SIZE,
        'device'               : str(device),
    }, cfg.OUTPUT_DIR)

    pd.DataFrame(history).to_csv(
        os.path.join(cfg.OUTPUT_DIR, 'training_history.csv'), index=False
    )
    print(f'\nArquivos salvos em: {cfg.OUTPUT_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treina detector de AMD no dataset HYAMD')
    parser.add_argument(
        '--arch',
        default=cfg.ARCH,
        choices=['mobilenetv3', 'efficientnet_lite0', 'convnext_femto', 'ibot'],
        help='Arquitetura do backbone (padrão: %(default)s)',
    )
    args = parser.parse_args()
    main(args.arch)
