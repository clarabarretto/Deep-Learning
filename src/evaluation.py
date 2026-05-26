"""
Métricas de avaliação e visualizações.

Implementa o protocolo bootstrap AUROC do artigo de referência
(Cohen et al., 2025): 1.000 amostras sem reposição usando 80% do
conjunto de teste, reportando média e desvio padrão.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auroc(model: nn.Module, loader, device: torch.device) -> float:
    """AUROC pontual sobre um DataLoader completo."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)).squeeze(1)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())
    return roc_auc_score(all_labels, all_probs)


def bootstrap_auroc(y_true: np.ndarray, y_prob: np.ndarray,
                    n_bootstrap: int = 1000, sample_frac: float = 0.8,
                    seed: int = 42):
    """
    Bootstrap AUROC seguindo o protocolo exato de Cohen et al. (2025):
    amostras sem reposição, 80% do conjunto de teste por iteração.

    Retorna (mean, std).
    """
    rng    = np.random.RandomState(seed)
    n      = len(y_true)
    k      = int(n * sample_frac)
    aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=k, replace=False)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return float(np.mean(aurocs)), float(np.std(aurocs))


def plot_training_curves(history: dict, best_val_auroc: float,
                         arch: str, output_dir: str):
    """Curvas de loss de treino e AUROC de validação por época."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['epoch'], history['train_loss'], 'b-o', markersize=4)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss (BCE)')
    axes[0].set_title('Loss de Treinamento')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['epoch'], history['val_auroc'], 'g-o', markersize=4)
    axes[1].axhline(y=best_val_auroc, color='r', linestyle='--',
                    alpha=0.5, label=f'Melhor: {best_val_auroc:.4f}')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('AUROC')
    axes[1].set_title('AUROC de Validação')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.5, 1.0])

    plt.suptitle(f'Curvas de Treinamento — {arch}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_and_probs(all_labels: np.ndarray, all_probs: np.ndarray,
                       bs_mean: float, bs_std: float,
                       arch: str, output_dir: str):
    """Curva ROC e distribuição de probabilidades previstas."""
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    fig, axes   = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(fpr, tpr, 'b-', lw=2,
                 label=f'{arch} (AUROC = {bs_mean:.3f} ± {bs_std:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    axes[0].fill_between(fpr, tpr, alpha=0.1)
    axes[0].set_xlabel('FPR (1 − Especificidade)')
    axes[0].set_ylabel('TPR (Sensibilidade)')
    axes[0].set_title(f'Curva ROC — {arch}')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(all_probs[all_labels == 0], bins=30, alpha=0.6,
                 label='non-AMD', color='steelblue', density=True)
    axes[1].hist(all_probs[all_labels == 1], bins=30, alpha=0.6,
                 label='AMD (moderado-tardio)', color='tomato', density=True)
    axes[1].axvline(x=0.5, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Probabilidade prevista de AMD')
    axes[1].set_ylabel('Densidade')
    axes[1].set_title('Distribuição de Probabilidades')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Avaliação Final — {arch}  (AUROC = {bs_mean:.3f} ± {bs_std:.3f})')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve_and_probs.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_auroc_by_age(results_df, non_amd_mask, output_dir: str) -> dict:
    """AUROC por faixa etária — reproduz a Seção 4 do artigo de referência."""
    age_bins   = [(18, 70), (70, 80), (80, 120)]
    age_labels = ['18-70', '70-80', '>80']
    auroc_age  = {}

    for (lo, hi), lab in zip(age_bins, age_labels):
        in_bin = (
            (results_df['AMD'] == 2) &
            (results_df['age'] >= lo) &
            (results_df['age'] < hi)
        )
        subset = results_df[non_amd_mask | in_bin].copy()
        subset['bin'] = in_bin[subset.index].astype(int)
        if subset['bin'].sum() > 0 and subset['bin'].nunique() == 2:
            auc = roc_auc_score(subset['bin'], subset['pred_prob'])
            auroc_age[lab] = auc
            n_amd     = int(subset['bin'].sum())
            n_non_amd = int((subset['bin'] == 0).sum())
            print(f'  Idade {lab:>5} | AMD={n_amd} vs non-AMD={n_non_amd} | AUROC = {auc:.3f}')
        else:
            print(f'  Idade {lab:>5} | amostras insuficientes')

    if auroc_age:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(list(auroc_age.keys()), list(auroc_age.values()),
               color=['#4878CF', '#6ACC65', '#D65F5F'], width=0.5)
        ax.set_ylim([0.5, 1.0])
        ax.set_xlabel('Faixa Etária AMD')
        ax.set_ylabel('AUROC')
        ax.set_title('AUROC por Faixa Etária')
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(auroc_age.values()):
            ax.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/auroc_by_age.png', dpi=150, bbox_inches='tight')
        plt.show()

    return auroc_age


def save_summary(summary: dict, output_dir: str):
    """Salva o resumo de resultados em JSON e imprime no console."""
    path = f'{output_dir}/results_summary.json'
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('=' * 55)
    print('         RESUMO FINAL DOS RESULTADOS')
    print('=' * 55)
    for k, v in summary.items():
        print(f'  {k:<35} {v}')
    print('=' * 55)
    print(f'\nSalvo em: {path}')
