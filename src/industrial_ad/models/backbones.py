from __future__ import annotations
import torch
import timm
from typing import Tuple, Literal
from torchvision import transforms

BackboneName = Literal["vit_tiny_patch16_224", "vit_small_patch16_224", "swin_tiny_patch4_window7_224"]

def get_backbone(
    name: BackboneName = "swin_tiny_patch4_window7_224",
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Devuelve:
      - model (eval, sin classifier)
      - preprocess torchvision.transforms
      - fn_features: (B,3,H,W)-> (B, N_patches, D), shape_parches=(Ph, Pw)
    """
    model = timm.create_model(name, pretrained=True, num_classes=0)  # headless
    model.to(device).eval()

    # Normalización ImageNet
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    @torch.no_grad()
    def fn_features(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        x: (B,3,H,W) en [0,1], no normalizado.
        retorna:
          feats: (B, N, D)
          grid: (Ph, Pw)
        """
        # aplicamos la misma normalización que 'preprocess' (ya suponemos resize/crop 224)
        # aquí solo normalizamos (si ya viene normalizado, omitir esta fn y usar model(x))
        raise_if_not_224 = (x.shape[-1] == 224 and x.shape[-2] == 224)
        if not raise_if_not_224:
            # asegurar 224
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Normalizar
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        xn = (x - mean) / std

        out = model.forward_features(xn)

        # Para ViT: out = (B, tokens, D). Eliminar CLS (token 0) y reconstruir grid.
        if out.ndim == 3:
            B, T, D = out.shape
            # intentar inferir grid cuadrícula
            # ViT patch=16 -> 14x14 tokens (196). Si hay CLS: T = 197.
            has_cls = (T in (197, 577, 145))
            if has_cls:
                out = out[:, 1:, :]  # quitar CLS
                T = out.shape[1]
            # cuadrícula
            g = int(T ** 0.5)
            feats = out
            grid = (g, g)
            return feats, grid

        # Para Swin: out = (B, C, Hf, Wf). Aplanamos a parches Hw*Ww
        elif out.ndim == 4:
            B, C, Hf, Wf = out.shape
            feats = out.permute(0,2,3,1).reshape(B, Hf*Wf, C).contiguous()
            return feats, (Hf, Wf)

        else:
            raise RuntimeError(f"Salida inesperada de forward_features: {out.shape}")

    return model, preprocess, fn_features
