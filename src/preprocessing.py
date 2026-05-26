"""
Pipeline de pré-processamento de imagens de fundo de olho.

Replica a Seção 3 de Cohen et al. (2025):
  1. Crop do fundo preto via threshold em escala de cinza
  2. Padding simétrico para imagem quadrada (preserva aspect ratio)
  3. Resize bilinear para target_size × target_size
"""

import numpy as np
from PIL import Image, ImageOps


def crop_fundus_background(img: Image.Image, threshold: int = 10) -> Image.Image:
    """Remove a borda preta não informativa presente em fotografias de fundo de olho."""
    gray = np.array(img.convert('L'))
    mask = gray > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    margin = 5
    rmin = max(0, rmin - margin)
    rmax = min(gray.shape[0], rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(gray.shape[1], cmax + margin)
    return img.crop((cmin, rmin, cmax, rmax))


def pad_to_square(img: Image.Image, fill: int = 0) -> Image.Image:
    """Padding simétrico com zeros para tornar a imagem quadrada."""
    w, h    = img.size
    max_dim = max(w, h)
    pad_w   = (max_dim - w) // 2
    pad_h   = (max_dim - h) // 2
    padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
    return ImageOps.expand(img, border=padding, fill=fill)


def preprocess_fundus(img: Image.Image, target_size: int = 224) -> Image.Image:
    """Pipeline completo: crop → pad → resize."""
    img = img.convert('RGB')
    img = crop_fundus_background(img)
    img = pad_to_square(img)
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return img
