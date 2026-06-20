"""Paso 2 del flujo: correr un detector publico de deepfakes sobre las imagenes.

Lee imagenes de data/images/real/ y data/images/fake/, corre un clasificador de
imagen (Hugging Face) y escribe data/preds.csv con la probabilidad de que cada
imagen sea fake.

Uso:
    python src/run_detector.py --model prithivMLmods/Deep-Fake-Detector-Model
    python src/run_detector.py --mock   # sin descargar nada, para probar el armazon
"""

import argparse
import os
import random
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Palabras que, si aparecen en la etiqueta del modelo, indican clase "fake".
FAKE_LABEL_HINTS = ("fake", "deepfake", "deep fake", "ai", "synthetic", "fauxtography")
# Palabras que indican clase "real".
REAL_LABEL_HINTS = ("real", "realism", "authentic", "genuine", "human")


def find_images(images_dir: Path):
    """Devuelve lista de (path, true_label) para real/ y fake/."""
    rows = []
    for true_label in ("real", "fake"):
        subdir = images_dir / true_label
        if not subdir.is_dir():
            continue
        for path in sorted(subdir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTS:
                rows.append((str(path), true_label))
    return rows


def label_to_p_fake(label: str, score: float) -> float:
    """Convierte una etiqueta+score del clasificador a probabilidad de fake."""
    norm = label.strip().lower()
    if any(hint in norm for hint in FAKE_LABEL_HINTS):
        return score
    if any(hint in norm for hint in REAL_LABEL_HINTS):
        return 1.0 - score
    # Etiqueta desconocida: avisamos y asumimos que el score es de "fake".
    return score


def scores_to_p_fake(predictions) -> float:
    """Dado el output de la pipeline (lista de {label, score}), calcula p_fake.

    Tomamos el aporte mas claro: si hay una etiqueta fake explicita usamos su
    score; si solo hay etiqueta real, usamos 1 - score. Como respaldo usamos la
    prediccion top-1.
    """
    for pred in predictions:
        norm = pred["label"].strip().lower()
        if any(hint in norm for hint in FAKE_LABEL_HINTS):
            return float(pred["score"])
    for pred in predictions:
        norm = pred["label"].strip().lower()
        if any(hint in norm for hint in REAL_LABEL_HINTS):
            return 1.0 - float(pred["score"])
    top = predictions[0]
    return label_to_p_fake(top["label"], float(top["score"]))


def run_mock(rows):
    """Genera p_fake aleatorio sin cargar ningun modelo."""
    rng = random.Random(42)
    out = []
    for path, true_label in rows:
        p_fake = rng.random()
        out.append((path, true_label, p_fake))
    return out


def run_real(rows, model_name: str, threshold: float):
    """Carga la pipeline de transformers y corre el detector imagen por imagen."""
    from PIL import Image
    from transformers import pipeline

    clf = pipeline("image-classification", model=model_name, top_k=None)
    out = []
    for path, true_label in rows:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # imagen corrupta o ilegible
            print(f"[warn] no se pudo abrir {path}: {exc}")
            continue
        preds = clf(image)
        p_fake = scores_to_p_fake(preds)
        out.append((path, true_label, p_fake))
    return out


def main():
    parser = argparse.ArgumentParser(description="Correr detector de deepfakes -> data/preds.csv")
    parser.add_argument("--model", default="prithivMLmods/Deep-Fake-Detector-Model",
                        help="Modelo de image-classification en Hugging Face.")
    parser.add_argument("--images", default="data/images",
                        help="Carpeta con subcarpetas real/ y fake/.")
    parser.add_argument("--out", default="data/preds.csv", help="CSV de salida.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Umbral de p_fake para marcar pred_label=fake.")
    parser.add_argument("--mock", action="store_true",
                        help="No carga el modelo; genera p_fake aleatorio.")
    args = parser.parse_args()

    images_dir = Path(args.images)
    rows = find_images(images_dir)
    if not rows:
        raise SystemExit(
            f"No se encontraron imagenes en {images_dir}/real y {images_dir}/fake. "
            "Coloca imagenes o usa --mock con archivos de prueba."
        )

    print(f"[info] {len(rows)} imagenes encontradas en {images_dir}")
    if args.mock:
        print("[info] modo --mock: p_fake aleatorio, sin cargar modelo")
        results = run_mock(rows)
        model_used = "mock"
    else:
        print(f"[info] cargando modelo {args.model} ...")
        results = run_real(rows, args.model, args.threshold)
        model_used = args.model

    df = pd.DataFrame(results, columns=["path", "true_label", "p_fake"])
    df["pred_label"] = (df["p_fake"] >= args.threshold).map({True: "fake", False: "real"})
    df["model"] = model_used

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[ok] escrito {out_path} ({len(df)} filas)")


if __name__ == "__main__":
    main()
