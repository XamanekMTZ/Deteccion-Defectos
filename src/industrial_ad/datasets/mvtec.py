from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from torch.utils.data import Dataset
import torch

@dataclass
class MVTecItem:
    img_path: Path
    mask_path: Optional[Path]  # None si es "good" (sin máscara)
    label: int                 # 0=good, 1=defect
    defect_type: str           # "good" o nombre del defecto
    category: str

class MVTecAD(Dataset):
    """
    Loader de MVTec AD.
    Estructura esperada:
      root/category/train/good/*.png
      root/category/test/{good,defect_type}/*.png
      root/category/ground_truth/defect_type/*.png
    """
    def __init__(
        self,
        root: str | Path,
        split: str = "train",            # "train" o "test"
        category: Optional[str] = None,  # si None, concatena todas las categorías
        img_size: int = 512,
        center_crop: Optional[int] = None,
        to_tensor: bool = True,
        return_mask: bool = True,
    ):
        self.root = Path(root)
        assert self.root.exists(), f"No existe la ruta {self.root}"
        assert split in {"train", "test"}, "split debe ser 'train' o 'test'"
        self.split = split
        self.img_size = img_size
        self.center_crop = center_crop
        self.to_tensor = to_tensor
        self.return_mask = return_mask

        self.categories = (
            [category] if category else sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        )
        assert len(self.categories) > 0, f"No hay categorías en {self.root}"

        self.items: List[MVTecItem] = []
        for cat in self.categories:
            cat_dir = self.root / cat
            if split == "train":
                # solo "good" en train
                imgs = sorted((cat_dir / "train" / "good").glob("*"))
                for ip in imgs:
                    if ip.is_file():
                        self.items.append(MVTecItem(
                            img_path=ip, mask_path=None, label=0, defect_type="good", category=cat
                        ))
            else:
                # test: good y defect types
                test_dir = cat_dir / "test"
                for defect_folder in sorted([p for p in test_dir.iterdir() if p.is_dir()]):
                    dname = defect_folder.name
                    imgs = sorted(defect_folder.glob("*"))
                    for ip in imgs:
                        if not ip.is_file():
                            continue
                        if dname == "good":
                            self.items.append(MVTecItem(
                                img_path=ip, mask_path=None, label=0, defect_type="good", category=cat
                            ))
                        else:
                            # máscara (si existe)
                            gt = cat_dir / "ground_truth" / dname / f"{ip.stem}_mask.png"
                            mask_path = gt if gt.exists() else None
                            self.items.append(MVTecItem(
                                img_path=ip, mask_path=mask_path, label=1, defect_type=dname, category=cat
                            ))

    def __len__(self) -> int:
        return len(self.items)

    def _read_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_mask(self, path: Optional[Path], size: Tuple[int,int]) -> np.ndarray:
        if path is None:
            return np.zeros(size, dtype=np.uint8)
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            # algunas categorías pueden carecer de máscara aun siendo defecto
            return np.zeros(size, dtype=np.uint8)
        return m

    def _resize_and_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        # Resize preservando aspecto al lado menor = img_size (simple y estable)
        scale = self.img_size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        if self.center_crop:
            ch = cw = self.center_crop
            y0 = max(0, (nh - ch) // 2)
            x0 = max(0, (nw - cw) // 2)
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
            "image_np": img,      # numpy RGB
            "mask_np": mask,      # numpy [0..255]
            "label": it.label,    # 0 good / 1 defect
            "defect_type": it.defect_type,
            "category": it.category,
            "path": str(it.img_path),
        }

        if self.to_tensor:
            # [H,W,3] -> [3,H,W], normalizado a [0,1]
            img_t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
            mask_t = torch.from_numpy((mask > 127).astype(np.float32))  # binaria
            sample.update({"image": img_t, "mask": mask_t})

        if not self.return_mask:
            sample.pop("mask_np", None)
            sample.pop("mask", None)

        return sample
