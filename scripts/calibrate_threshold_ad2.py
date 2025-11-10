import argparse, json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from industrial_ad.datasets import MVTecAD2
from industrial_ad.models.backbones import get_backbone
from industrial_ad.anomaly.patchcore_simple import MemoryBank

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--backbone", default="swin_tiny_patch4_window7_224",
                    choices=["vit_tiny_patch16_224","vit_small_patch16_224","swin_tiny_patch4_window7_224"])
    ap.add_argument("--percentile", type=float, default=99.5,
                    help="Percentil de scores en VALIDATION para fijar τ (ej. 99.5)")
    ap.add_argument("--out", default="outputs/thresholds.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, fn_features = get_backbone(args.backbone, device=device)

    # carga banco
    ckpt = torch.load(args.bank, map_location="cpu")
    bank_t = ckpt["bank"]
    mb = MemoryBank(max_patches=bank_t.shape[0], normalize=True, metric="cosine")
    mb.build([bank_t])

    # dataset VALIDATION (solo normales)
    ds_val = MVTecAD2(root=args.root, split="validation", category=args.category, img_size=256)

    def to_224(x):
        return torch.nn.functional.interpolate(x.unsqueeze(0), size=(224,224), mode="bilinear", align_corners=False)[0]

    img_scores = []
    for i in tqdm(range(len(ds_val)), desc="Calibrating"):
        s = ds_val[i]
        x224 = to_224(s["image"]).unsqueeze(0).to(device)
        feats, grid = fn_features(x224)
        ps = mb.knn_score(feats[0].cpu(), k=5)  # score por parche
        img_scores.append(float(ps.max().item()))  # score imagen = max parche

    if len(img_scores) == 0:
        raise RuntimeError("VALIDATION vacío; no se puede calibrar τ")

    tau = float(np.percentile(np.array(img_scores), args.percentile))
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # merge/update thresholds
    data = {}
    if outp.exists():
        try:
            data = json.loads(outp.read_text())
        except Exception:
            pass
    key = f"{args.category}:{args.backbone}"
    data[key] = {"tau": tau, "percentile": args.percentile}
    outp.write_text(json.dumps(data, indent=2))
    print(f"[OK] τ={tau:.6f} guardado en {outp} clave='{key}'")

if __name__ == "__main__":
    main()
