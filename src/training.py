"""
Utilitários de treinamento: scheduler com warmup linear + decaimento coseno,
loop de uma época e cálculo de loss de validação.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm


def get_scheduler_with_warmup(optimizer: optim.Optimizer,
                               warmup_steps: int,
                               total_steps: int) -> optim.lr_scheduler.LambdaLR:
    """Warmup linear seguido de decaimento coseno até 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model: nn.Module,
                    loader,
                    optimizer: optim.Optimizer,
                    scheduler,
                    criterion: nn.Module,
                    device: torch.device) -> float:
    """Uma época de treino com gradient clipping. Retorna a loss média."""
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, desc='Treino', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs).squeeze(1), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def compute_val_loss(model: nn.Module,
                     loader,
                     criterion: nn.Module,
                     device: torch.device) -> float:
    """Loss de validação sem gradiente, normalizada pelo número de amostras."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            total_loss += criterion(model(imgs).squeeze(1), labels).item() * len(labels)
    return total_loss / len(loader.dataset)
