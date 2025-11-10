# scripts/infer_dataset_ad2.py
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
from tqdm import tqdm
from industrial_ad.datasets import MVTecAD2
from industrial_ad.models.backbones import get_backbone
from industrial_ad.anomaly.patchcore_simple import MemoryBank, aggregate_patch_scores

def overlay_and_save(rgb, heat, out_path, thr=0.6):
    heat_u8 = (np.clip(heat,0,1)*255).astype(np.uint8)
    heat_col = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_col = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 0.65, heat_col, 0.35, 0)

    mask = (heat > thr).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (255,255,255), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--backbone", default="swin_tiny_patch4_window7_224",
                    choices=["vit_tiny_patch16_224","vit_small_patch16_224","swin_tiny_patch4_window7_224"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--thresholds", default="outputs/thresholds.json")
    ap.add_argument("--tau", type=float, default=None, help="Umbral manual (si no hay thresholds.json)")
    ap.add_argument("--save-fp", action="store_true", help="Guardar también buenas con score>=tau (posibles FP)")
    ap.add_argument("--outdir", default="outputs/infer_ds")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, fn_features = get_backbone(args.backbone, device=device)

    # cargar banco
    ckpt = torch.load(args.bank, map_location="cpu")
    bank_t = ckpt["bank"]
    mb = MemoryBank(max_patches=bank_t.shape[0], normalize=True, metric="cosine")
    mb.build([bank_t])

    # τ
    tau = args.tau
    key = f"{args.category}:{args.backbone}"
    if tau is None and Path(args.thresholds).exists():
        try:
            d = json.loads(Path(args.thresholds).read_text())
            if key in d:
                tau = float(d[key]["tau"])
        except Exception:
            pass
    if tau is None:
        raise SystemExit("No hay τ. Usa --tau o ejecuta primero calibrate_threshold_ad2.py")

    # dataset: test_public (tiene GT)
    ds = MVTecAD2(root=args.root, split="test_public", category=args.category, img_size=256)
    outdir = Path(args.outdir) / args.category
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results.csv"

    def to_224(x):
        return torch.nn.functional.interpolate(x.unsqueeze(0), size=(224,224), mode="bilinear", align_corners=False)[0]

    rows = ["path,label,defect_type,score,decision"]
    for i in tqdm(range(len(ds)), desc="Infer test_public"):
        s = ds[i]
        x224 = to_224(s["image"]).unsqueeze(0).to(device)
        feats, grid = fn_features(x224)
        ps = mb.knn_score(feats[0].cpu(), k=args.k)
        score = float(ps.max().item())
        decision = "NG" if score >= tau else "OK"
        rows.append(f"{s['path']},{s['label']},{s['defect_type']},{score:.6f},{decision}")

        # generar overlay si:
        #   a) imagen está etiquetada como defecto
        #   b) es 'good' pero score>=tau y --save-fp activado
        save_this = (s["label"] == 1) or (args.save_fp and score >= tau and s["label"] == 0)
        if save_this:
            H,W = s["image_np"].shape[:2]
            heat = aggregate_patch_scores(ps, grid, (H,W)).numpy().astype(np.float32)
            # normalización robusta + suavizado
            p2,p98 = np.percentile(heat, [2.0,98.0])
            heat = np.clip((heat - p2)/max(p98-p2,1e-6), 0, 1)
            heat = cv2.GaussianBlur(heat, (5,5), 0)

            tag = f"{'DEF' if s['label']==1 else 'FP'}_{s['defect_type']}"
            fname = f"{Path(s['path']).stem}_{tag}_{decision}_{score:.4f}.png"
            overlay_and_save(s["image_np"], heat, outdir / fname, thr=0.6)

    csv_path.write_text("\n".join(rows))
    print(f"[OK] CSV → {csv_path}")
    print(f"[OK] Overlays → {outdir}  (defectos y {'FP' if args.save_fp else 'sin FP'})")

if __name__ == "__main__":
    main()
