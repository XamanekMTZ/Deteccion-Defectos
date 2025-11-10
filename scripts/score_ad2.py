import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import cv2

from industrial_ad.datasets import MVTecAD2
from industrial_ad.models.backbones import get_backbone
from industrial_ad.anomaly.patchcore_simple import MemoryBank, aggregate_patch_scores

# def save_heatmap(rgb_np, heat_np, out_path):
#     # rgb_np: H,W,3 [0..255]; heat_np: H,W [0..1]
#     plt.figure()
#     plt.imshow(rgb_np.astype(np.uint8))
#     plt.imshow(heat_np, alpha=0.5)
#     plt.axis("off")
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
#     plt.close()


# def save_heatmap(rgb_np, heat_np, out_path):
#     """
#     rgb_np: imagen RGB uint8 (H,W,3)
#     heat_np: mapa [0..1] float32 (H,W)
#     out_path: destino .png
#     """
#     # Convertir heatmap a colormap tipo "jet" (azul->rojo)
#     heat_col = cv2.applyColorMap((heat_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
#     heat_col = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)

#     # Mezclar con la imagen original (alpha más bajo)
#     overlay = cv2.addWeighted(rgb_np, 0.6, heat_col, 0.4, 0)

#     # Borde blanco opcional donde la anomalía > 0.6
#     mask = (heat_np > 0.6).astype(np.uint8)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def save_heatmap(rgb_np, heat_np, out_path):
    # heat_np ∈ [0,1]  → colormap tipo JET
    heat_u8 = (np.clip(heat_np, 0, 1) * 255).astype(np.uint8)
    heat_col = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_col = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)

    # mezcla suave
    overlay = cv2.addWeighted(rgb_np, 0.65, heat_col, 0.35, 0)

    # contornos opcionales (umbral alto = 0.6)
    mask = (heat_np > 0.6).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--bank", required=True, help="outputs/banks/<cat>/memory_bank.pt")
    ap.add_argument("--backbone", default="swin_tiny_patch4_window7_224",
                    choices=["vit_tiny_patch16_224","vit_small_patch16_224","swin_tiny_patch4_window7_224"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--outdir", default="outputs/eval")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, fn_features = get_backbone(args.backbone, device=device)

    # cargar banco
    ckpt = torch.load(args.bank, map_location="cpu")
    bank_t = ckpt["bank"]  # (M,D)
    mbank = MemoryBank(max_patches=bank_t.shape[0], normalize=True, metric="cosine")
    mbank.build([bank_t])

    # test público
    ds_test = MVTecAD2(root=args.root, split="test_public", category=args.category, img_size=256)

    img_scores, img_labels = [], []
    all_pix_scores, all_pix_gts = [], []

    def to_224(img_t):  # (3,H,W)->(1,3,224,224)
        return torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(224,224), mode="bilinear", align_corners=False)[0]

    for i in tqdm(range(len(ds_test)), desc="Scoring"):
        s = ds_test[i]
        x = s["image"].to(device)               # [0,1]
        x224 = to_224(x).unsqueeze(0)           # (1,3,224,224)

        feats, grid = fn_features(x224)         # (1, N, D)
        feats = feats[0].cpu()
        # kNN sobre banco
        patch_scores = mbank.knn_score(feats, k=args.k)  # (N,)
        # score de imagen = max por parche
        img_score = float(patch_scores.max().item())

        img_scores.append(img_score)
        img_labels.append(int(s["label"]))      # 0 good / 1 defect

        # mapa de calor (0..1)
        H, W = s["image_np"].shape[:2]
        heat = aggregate_patch_scores(patch_scores, grid, (H, W))

        # normalización robusta: recorta percentiles 2–98 para evitar saturaciones
        h = heat.numpy().astype(np.float32)
        p2, p98 = np.percentile(h, [2.0, 98.0])
        h = np.clip((h - p2) / max(p98 - p2, 1e-6), 0, 1)

        # suavizado leve (quita “bandas” verticales por ventanas de Swin)
        h = cv2.GaussianBlur(h, (5, 5), 0)

        # limpia ruido fino
        h = cv2.morphologyEx(h, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        #heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        #all_pix_scores.append(heat.numpy().astype(np.float32))
        all_pix_scores.append(h.astype(np.float32))

        # GT píxel (0/1)
        gt = (s["mask_np"] > 127).astype(np.uint8)
        all_pix_gts.append(gt)

        # guardar algunas visualizaciones
        if i < 8:
            out_img = Path(args.outdir) / args.category / "viz" / f"{i:03d}.png"
            save_heatmap(s["image_np"], heat.numpy(), out_img)

    # Métricas imagen
    auroc_img = roc_auc_score(img_labels, img_scores)
    aupr_img  = average_precision_score(img_labels, img_scores)

    # Métrica simple píxel (AUROC-pixel). Nota: PRO requiere barrido con IoU; aquí damos AUROC-pixel base.
    pix_scores_flat = np.concatenate([p.flatten() for p in all_pix_scores], axis=0)
    pix_gts_flat    = np.concatenate([g.flatten() for g in all_pix_gts], axis=0)
    auroc_pix = roc_auc_score(pix_gts_flat, pix_scores_flat)

    outdir = Path(args.outdir) / args.category
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.txt").write_text(
        f"Category: {args.category}\nBackbone: {args.backbone}\n"
        f"AUROC_image: {auroc_img:.4f}\nAUPR_image: {aupr_img:.4f}\nAUROC_pixel: {auroc_pix:.4f}\n"
        f"Samples: {len(ds_test)}\n"
    )
    print(f"[OK] AUROC_image={auroc_img:.4f} | AUPR_image={aupr_img:.4f} | AUROC_pixel={auroc_pix:.4f}")
    print(f"Resultados en: {outdir}/results.txt  | Visualizaciones en {outdir}/viz/")

if __name__ == "__main__":
    main()
