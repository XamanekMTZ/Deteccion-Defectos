from __future__ import annotations
import cv2, numpy as np, torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from torch.utils.data import Dataset

@dataclass
class AD2Item:
    img_path: Path
    mask_path: Optional[Path]  # solo existe públicamente en test_public
    label: int                 # 0=normal, 1=defecto
    split: str                 # train/validation/test_public/test_private/test_private_mixed
    category: str
    defect_type: str           # 'good' u otro nombre

class MVTecAD2(Dataset):
    """
    Estructura (según página oficial):
      category/
        train/ (solo normales)
        validation/ (solo normales)
        test_public/{good,defect_type}/*.png  + ground_truth/{defect_type}/*_mask.png
        test_private/{good,defect_type}/*.png
        test_private_mixed/{good,defect_type}/*.png
    Nota: máscaras públicas SOLO para test_public. (private y private_mixed no exponen GT pública)
    """
    VALID_SPLITS = {"train","validation","test_public","test_private","test_private_mixed"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        category: Optional[str] = None,
        img_size: int = 512,
        center_crop: Optional[int] = None,
        to_tensor: bool = True,
        return_mask: bool = True,
    ):
        self.root = Path(root)
        assert self.root.exists(), f"No existe la ruta {self.root}"
        assert split in self.VALID_SPLITS, f"split inválido: {split}"
        self.split = split
        self.img_size = img_size
        self.center_crop = center_crop
        self.to_tensor = to_tensor
        self.return_mask = return_mask

        self.categories = [category] if category else sorted(
            [p.name for p in self.root.iterdir() if p.is_dir()]
        )
        assert self.categories, f"No hay categorías en {self.root}"

        self.items: List[AD2Item] = []
        # for cat in self.categories:
        #     cat_dir = self.root / cat

        #     if split in {"train","validation"}:
        #         imgs = sorted((cat_dir / split).glob("*"))
        #         for ip in imgs:
        #             if ip.is_file():
        #                 self.items.append(AD2Item(
        #                     img_path=ip, mask_path=None, label=0,
        #                     split=split, category=cat, defect_type="good"
        #                 ))
        #     else:
        #         test_dir = cat_dir / split
        #         for ddir in sorted([p for p in test_dir.iterdir() if p.is_dir()]):
        #             dname = ddir.name  # 'good' o defecto
        #             for ip in sorted(ddir.glob("*")):
        #                 if not ip.is_file():
        #                     continue
        #                 if dname == "good":
        #                     self.items.append(AD2Item(
        #                         img_path=ip, mask_path=None, label=0,
        #                         split=split, category=cat, defect_type="good"
        #                     ))
        #                 else:
        #                     mask = None
        #                     if split == "test_public":
        #                         # máscaras disponibles y públicas en test_public
        #                         m = cat_dir / "ground_truth" / dname / f"{ip.stem}_mask.png"
        #                         mask = m if m.exists() else None
        #                     self.items.append(AD2Item(
        #                         img_path=ip, mask_path=mask, label=1,
        #                         split=split, category=cat, defect_type=dname
        #                     ))
        for cat in self.categories:
            cat_dir = self.root / cat
        
            if split in {"train","validation"}:
                imgs = sorted((cat_dir / split).rglob("*.png"))
                for ip in imgs:
                    if ip.is_file():
                        self.items.append(AD2Item(
                            img_path=ip, mask_path=None, label=0,
                            split=split, category=cat, defect_type="good"
                        ))
        
            else:
                test_dir = cat_dir / split
                # si hay subcarpetas por tipo ('good', 'crack', etc.)
                if test_dir.exists():
                    subdirs = [p for p in test_dir.iterdir() if p.is_dir()]
                else:
                    subdirs = []
        
                if subdirs:
                    # Estructura típica: test_public/good/*.png, test_public/<defecto>/*.png
                    for ddir in sorted(subdirs):
                        dname = ddir.name
                        imgs = sorted(ddir.rglob("*.png"))   # <-- rglob para robustez
                        for ip in imgs:
                            if not ip.is_file():
                                continue
                            if dname == "good":
                                self.items.append(AD2Item(
                                    img_path=ip, mask_path=None, label=0,
                                    split=split, category=cat, defect_type="good"
                                ))
                            else:
                                mask = None
                                if split == "test_public":
                                    m = cat_dir / "ground_truth" / dname / f"{ip.stem}_mask.png"
                                    mask = m if m.exists() else None
                                self.items.append(AD2Item(
                                    img_path=ip, mask_path=mask, label=1,
                                    split=split, category=cat, defect_type=dname
                                ))
                else:
                    # Caso raro: imágenes planas directamente en test_* sin subcarpetas
                    imgs = sorted(test_dir.rglob("*.png"))
                    for ip in imgs:
                        if not ip.is_file():
                            continue
                        # sin subcarpetas no podemos distinguir tipo; asumimos 'unknown'
                        mask = None
                        if split == "test_public":
                            m = cat_dir / "ground_truth" / f"{ip.stem}_mask.png"
                            mask = m if m.exists() else None
                        self.items.append(AD2Item(
                            img_path=ip, mask_path=mask, label=1,  # conservador
                            split=split, category=cat, defect_type="unknown"
                        ))

    def __len__(self) -> int:
        return len(self.items)

    def _read_image(self, p: Path) -> np.ndarray:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer {p}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_mask(self, p: Optional[Path], size_hw: tuple[int,int]) -> np.ndarray:
        if p is None:
            return np.zeros(size_hw, dtype=np.uint8)
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return np.zeros(size_hw, dtype=np.uint8)
        return m

    def _resize_and_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = self.img_size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        if self.center_crop:
            ch = cw = self.center_crop
            y0 = max(0, (nh - ch) // 2); x0 = max(0, (nw - cw) // 2)
            img = img[y0:y0+ch, x0:x0+cw]
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        img = self._read_image(it.img_path)
        img = self._resize_and_crop(img)
        h, w = img.shape[:2]

        mask = self._read_mask(it.mask_path, (h, w))
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        sample = {
            "image_np": img,
            "mask_np": mask,
            "image": torch.from_numpy(img).permute(2,0,1).float() / 255.0,
            "mask": torch.from_numpy((mask > 127).astype(np.float32)),
            "label": it.label,
            "split": it.split,
            "category": it.category,
            "defect_type": it.defect_type,
            "path": str(it.img_path),
        }

        if not self.return_mask:
            sample.pop("mask_np", None); sample.pop("mask", None)

        return sample
