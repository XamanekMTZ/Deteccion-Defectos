import argparse
from industrial_ad.datasets import MVTecAD2
from industrial_ad.config import make_default_config
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Ruta a data/mvtec_ad2")
    ap.add_argument("--category", default=None, help="p.ej. Fabric, Rice, Can, ...")
    ap.add_argument("--split", default="test_public",
                    choices=["train","validation","test_public","test_private","test_private_mixed"])
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--center-crop", type=int, default=None)
    args = ap.parse_args()

    cfg = make_default_config(root=args.root)
    ds = MVTecAD2(
        root=cfg.data.root, split=args.split, category=args.category,
        img_size=args.img_size, center_crop=args.center_crop,
        to_tensor=True, return_mask=True
    )
    print(f"[OK] AD2 cargado: split={args.split}, categorías={ds.categories}, total={len(ds)}")

    # muestreo breve
    for i in tqdm(range(min(5, len(ds))), desc="Preview"):
        s = ds[i]
        has_mask = "mask" in s
        print(f"{i}: cat={s['category']} split={s['split']} type={s['defect_type']} "
              f"label={s['label']} img={tuple(s['image'].shape)} mask={'yes' if has_mask else 'no'} "
              f"path={s['path']}")

if __name__ == "__main__":
    main()
