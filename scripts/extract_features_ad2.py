import argparse, json
from pathlib import Path
import torch
from tqdm import tqdm
from industrial_ad.datasets import MVTecAD2
from industrial_ad.models.backbones import get_backbone

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data/mvtec_ad2")
    ap.add_argument("--category", required=True, help="p.ej. Fabric")
    ap.add_argument("--backbone", default="swin_tiny_patch4_window7_224",
                    choices=["vit_tiny_patch16_224","vit_small_patch16_224","swin_tiny_patch4_window7_224"])
    ap.add_argument("--outdir", default="outputs/banks")
    ap.add_argument("--max-patches", type=int, default=20000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, fn_features = get_backbone(args.backbone, device=device)

    # dataset: solo normales -> train + validation
    ds_train = MVTecAD2(root=args.root, split="train", category=args.category, img_size=256, center_crop=None)
    ds_val   = MVTecAD2(root=args.root, split="validation", category=args.category, img_size=256, center_crop=None)

    def to_224(img_t):
        # img_t viene en [0,1], (3,H,W). Redimensionar a 224
        return torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(224,224), mode="bilinear", align_corners=False)[0]

    feats_all = []
    for split_name, ds in [("train", ds_train), ("validation", ds_val)]:
        for i in tqdm(range(len(ds)), desc=f"Extract {split_name}"):
            s = ds[i]
            x = s["image"].to(device)           # [0,1]
            x224 = to_224(x).unsqueeze(0)       # (1,3,224,224)
            feats, grid = fn_features(x224)     # (1, N, D)
            feats_all.append(feats[0].cpu())

    # construir banco simple (submuestreo aleatorio)
    bank = torch.cat(feats_all, dim=0)
    if bank.shape[0] > args.max_patches:
        idx = torch.randperm(bank.shape[0])[:args.max_patches]
        bank = bank[idx]
    bank = torch.nn.functional.normalize(bank, dim=1).float()

    outdir = Path(args.outdir) / args.category
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"bank": bank, "backbone": args.backbone}, outdir / "memory_bank.pt")
    (outdir / "meta.json").write_text(json.dumps({
        "category": args.category,
        "n_patches": int(bank.shape[0]),
        "dim": int(bank.shape[1]),
        "backbone": args.backbone
    }, indent=2))
    print(f"[OK] Banco guardado en {outdir}/memory_bank.pt ({bank.shape[0]} parches, D={bank.shape[1]})")

if __name__ == "__main__":
    main()
