import argparse
from pathlib import Path
from industrial_ad.datasets import MVTecAD
from industrial_ad.config import make_default_config
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Ruta a data/mvtec_ad")
    ap.add_argument("--category", type=str, default=None, help="Ej: carpet, grid, leather, ...")
    ap.add_argument("--split", type=str, default="train", choices=["train","test"])
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--center-crop", type=int, default=None)
    args = ap.parse_args()

    cfg = make_default_config(root=args.root)
    cfg.data.category = args.category
    cfg.data.img_size = args.img_size
    cfg.data.center_crop = args.center_crop

    ds = MVTecAD(
        root=cfg.data.root,
        split=args.split,
        category=cfg.data.category,
        img_size=cfg.data.img_size,
        center_crop=cfg.data.center_crop,
        to_tensor=True,
        return_mask=True
    )
    print(f"[OK] Dataset cargado: split={args.split}, categorías={ds.categories}, total={len(ds)}")

    # inspecciona 5 muestras
    for i in tqdm(range(min(5, len(ds))), desc="Preview"):
        s = ds[i]
        msg = f"{i}: cat={s['category']}, defect_type={s['defect_type']}, label={s['label']}, img={s['image'].shape}"
        if 'mask' in s:
            msg += f", mask={s['mask'].shape}, path={s['path']}"
        print(msg)

if __name__ == "__main__":
    main()
