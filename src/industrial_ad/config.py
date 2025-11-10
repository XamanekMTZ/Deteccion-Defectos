from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataConfig:
    root: Path                  # Ruta a data/mvtec_ad
    category: str | None = None # p.ej. "carpet", "grid", etc.
    img_size: int = 512         # Tamaño al que se redimensiona
    center_crop: int | None = None  # Opcional

@dataclass
class RuntimeConfig:
    num_workers: int = 4
    seed: int = 42

@dataclass
class Paths:
    outputs: Path
    logs: Path

@dataclass
class Config:
    data: DataConfig
    runtime: RuntimeConfig
    paths: Paths

def make_default_config(root: str = "data/mvtec_ad") -> Config:
    root_path = Path(root).resolve()
    outputs = Path("outputs").resolve()
    logs = outputs / "logs"
    return Config(
        data=DataConfig(root=root_path),
        runtime=RuntimeConfig(),
        paths=Paths(outputs=outputs, logs=logs),
    )
