"""
Mapas de explicabilidade para classificação de AMD.

Duas implementações conforme o tipo de arquitetura:

  AttentionCAM_ViT
    Atenção CLS-to-patch do último bloco transformer, média entre todas as
    cabeças. Implementa a visualização descrita na Seção III-G do relatório
    e nas Figuras 5 e 6.

  GradCAM_CNN
    Grad-CAM clássico (Selvaraju et al., 2020) para CNNs do timm.
    Hookeia o último bloco de features de cada arquitetura suportada.

Ambas retornam um mapa espacial normalizado em [0, 1], compatível com
overlay_attention para visualização.
"""

import numpy as np
import torch
from PIL import Image


class AttentionCAM_ViT:
    """
    Mapa de atenção CLS-to-patch do último bloco ViT.

    O hook é registrado no attn_drop (aplicado à matriz de atenção antes da
    ponderação dos valores), então inp[0] é o tensor (B, heads, N, N) após
    softmax — sem necessidade de gradiente.

    N = num_patches + 1 (índice 0 = token CLS).
    """

    def __init__(self, model):
        self.model = model
        self._attn = None
        self._handle = model.blocks[-1].attn.attn_drop.register_forward_hook(
            lambda m, inp, out: setattr(self, "_attn", inp[0].detach())
        )

    def remove_hook(self):
        self._handle.remove()

    def generate(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Retorna mapa de atenção espacial (h_patches, w_patches) ∈ [0, 1].
        img_tensor: (1, C, H, W) já no device correto.
        """
        self.model.eval()
        with torch.no_grad():
            self.model(img_tensor)

        attn = self._attn  # (1, heads, N, N)
        cls_attn = attn[0, :, 0, 1:]  # (heads, patches) — CLS → patches
        cls_attn = cls_attn.mean(0).cpu().numpy()  # (patches,) — média entre cabeças

        n = cls_attn.shape[0]
        h = w = int(n**0.5)
        cam = cls_attn.reshape(h, w)

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam


class GradCAM_CNN:
    """
    Grad-CAM genérico para CNNs do timm.

    Hookeia automaticamente o último bloco de features de cada arquitetura:
      mobilenetv3, efficientnet_lite0 → model.blocks[-1]
      convnext_femto                  → model.stages[-1]
    """

    _LAYER_MAP = {
        "mobilenetv3": lambda m: m.blocks[-1],
        "efficientnet_lite0": lambda m: m.blocks[-1],
        "convnext_femto": lambda m: m.stages[-1],
    }

    def __init__(self, model, arch: str):
        if arch not in self._LAYER_MAP:
            raise ValueError(
                f"Arquitetura '{arch}' não suportada no GradCAM_CNN. "
                f"Opções: {list(self._LAYER_MAP)}"
            )
        self.model = model
        self.gradients = None
        self.activations = None
        self._handles = []

        layer = self._LAYER_MAP[arch](model)
        self._handles.append(
            layer.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o.detach())
            )
        )
        self._handles.append(
            layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0].detach())
            )
        )

    def remove_hooks(self):
        for h in self._handles:
            h.remove()

    def generate(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Retorna mapa Grad-CAM (H', W') ∈ [0, 1]."""
        self.model.eval()
        self.model.zero_grad()

        with torch.enable_grad():
            logit = self.model(img_tensor).squeeze()
            logit.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze()
        cam = cam.cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam


def overlay_attention(
    img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.6, bg_threshold: int = 10
) -> Image.Image:
    """
    Sobrepõe um heatmap CAM à imagem de fundo de olho, suprimindo o fundo preto.

    A normalização é calculada apenas dentro da área retinal (pixels acima de
    bg_threshold), impedindo que a borda preta distorça a escala do mapa.
    """
    import matplotlib.cm as cm

    img_np = np.array(img_pil)
    gray = np.mean(img_np, axis=2)
    fg_mask = gray > bg_threshold

    cam_resized = (
        np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                img_pil.size, Image.BILINEAR
            )
        )
        / 255.0
    )

    cam_fg = cam_resized.copy()
    if fg_mask.any():
        vmin, vmax = cam_resized[fg_mask].min(), cam_resized[fg_mask].max()
        if vmax > vmin:
            cam_fg[fg_mask] = (cam_resized[fg_mask] - vmin) / (vmax - vmin)
        cam_fg[~fg_mask] = 0.0

    heatmap = cm.jet(cam_fg)[:, :, :3]
    img_float = img_np / 255.0
    overlay = img_float.copy()
    overlay[fg_mask] = alpha * heatmap[fg_mask] + (1 - alpha) * img_float[fg_mask]
    return Image.fromarray((overlay * 255).astype(np.uint8))
