"""
Factory de modelos para classificação binária de AMD.

Todos os modelos retornam um único logit (BCEWithLogitsLoss).
Os pesos pré-treinados são carregados via timm com checkpoints ImageNet-1k.

Sobre a escolha do backbone (Seção III-E/F do relatório):
  - iBOT ViT-L/16 (307M params) é inviável com ~1.000 imagens HYAMD;
    o ratio parâmetros/amostras causa instabilidade de otimização.
  - ViT-S/16 DINO (22M params) preserva a premissa metodológica central do
    artigo de referência (SSL pretraining) dentro de limites computacionais
    viáveis. DINO e iBOT pertencem à mesma família de métodos SSL.
"""

from typing import Callable

import timm
import torch.nn as nn


def build_ibot_model(num_classes: int = 1, img_size: int = 224) -> nn.Module:
    """ViT-Small/16 com pretraining DINO — proxy escalado do iBOT ViT-L/16."""
    return timm.create_model(
        'vit_small_patch16_224.dino',
        pretrained=True,
        num_classes=num_classes,
        img_size=img_size,
        dynamic_img_size=True,
    )


def build_mobilenetv3_large(num_classes: int = 1) -> nn.Module:
    """MobileNetV3-Large pré-treinado no ImageNet-1k (~5.5M parâmetros)."""
    return timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=num_classes)


def build_efficientnet_lite0(num_classes: int = 1) -> nn.Module:
    """EfficientNet-Lite0 pré-treinado no ImageNet-1k (~4.7M parâmetros)."""
    return timm.create_model('efficientnet_lite0', pretrained=True, num_classes=num_classes)


def build_convnext_femto(num_classes: int = 1) -> nn.Module:
    """ConvNeXt-Femto pré-treinado no ImageNet-1k (~5.2M parâmetros)."""
    return timm.create_model('convnext_femto', pretrained=True, num_classes=num_classes)


ARCH_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    'ibot'              : build_ibot_model,
    'mobilenetv3'       : build_mobilenetv3_large,
    'efficientnet_lite0': build_efficientnet_lite0,
    'convnext_femto'    : build_convnext_femto,
}


def build_model(arch: str, **kwargs) -> nn.Module:
    """Constrói um modelo pelo nome da arquitetura definida em ARCH_REGISTRY."""
    if arch not in ARCH_REGISTRY:
        raise ValueError(f"Arquitetura '{arch}' não encontrada. Opções: {list(ARCH_REGISTRY)}")
    return ARCH_REGISTRY[arch](**kwargs)
