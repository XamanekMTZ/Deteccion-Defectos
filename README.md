# Inspección de defectos con ViT/Swin

## Requisitos
- Python 3.9+
- CUDA opcional (para entrenamiento/inferencia acelerada)

## Instalar dependencias
```bash
python -m venv .venv
source .venv/bin/activate # En Windows usar `.venv\Scripts\activate`
# O usar conda:
# conda create -n industrial_ad python=3.9
# conda activate industrial_ad
# O usar virtualenvwrapper
# mkvirtualenv industrial_ad --python=python3.9
pip install -U pip
pip install -e .
```

## Datos (MVTec AD)
Coloca el dataset descomprimido así:
```
data/mvtec_ad2/
Can/
Fabric/
...
...
```
## Verificar dataset
```bash
python scripts/verify_mvtec.py --data-path data/mvtec_ad
```

``` bash
# Verificar el loader MVTec AD2
python scripts/verify_mvtec_ad2.py --root data/mvtec_ad2 --category Fabric --split test_public
``` 
## Extracción de características y construcción del banco de memoria ViT
```bash
python scripts/extract_features_ad2.py \
  --root data/mvtec_ad2 \
  --category Fabric \
  --backbone vit_small_patch16_224 \
  --outdir outputs/banks \
  --max-patches 50000
```

## Calibrar umbral (threshold) con validation (normales)
```bash
python scripts/calibrate_threshold_ad2.py \
  --root data/mvtec_ad2 \
  --category Fabric \
  --bank outputs/banks/Fabric/memory_bank.pt \
  --backbone vit_small_patch16_224 \
  --percentile 99.5 \
  --out outputs/thresholds.json
```

## Evaluar en test_public (métricas + heatmaps) 
```bash
python scripts/score_ad2.py \
  --root data/mvtec_ad2 \
  --category Fabric \
  --backbone vit_small_patch16_224 \
  --bank outputs/banks/Fabric/memory_bank.pt \
  --outdir outputs/eval \
  --k 3
```

## Inferir en dataset completo y guardar solo defectuosos
```bash
# Solo defectuosas:
python scripts/infer_dataset_ad2.py \
  --root data/mvtec_ad2 \
  --category Fabric \
  --bank outputs/banks/Fabric/memory_bank.pt \
  --backbone vit_small_patch16_224 \
  --thresholds outputs/thresholds.json \
  --k 3 \
  --outdir outputs/infer_ds

# (Opcional) Guardar también buenas con score>=τ:
python scripts/infer_dataset_ad2.py \
  --root data/mvtec_ad2 \
  --category Fabric \
  --bank outputs/banks/Fabric/memory_bank.pt \
  --backbone vit_small_patch16_224 \
  --thresholds outputs/thresholds.json \
  --k 3 \
  --save-fp \
  --outdir outputs/infer_ds
```

