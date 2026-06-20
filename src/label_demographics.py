"""Paso 3 del flujo: etiquetar demografia percibida de cada rostro.

Usa deepface (que envuelve FairFace) para estimar edad, genero y raza de cada
imagen. Filtra menores (regla etica: solo adultos) y escribe data/labels.csv.

IMPORTANTE (etica): la demografia es PERCIBIDA por un modelo, no declarada por la
persona. Es un proxy y una limitacion; debe documentarse en la seccion de doble
uso del reporte.

Uso:
    python src/label_demographics.py --images data/images --min_age 20
    python src/label_demographics.py --mock   # sin descargar nada
"""

import argparse
import random
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Categorias de raza de FairFace/deepface (incluye "latino hispanic", clave LatAm).
RACE_CATEGORIES = [
    "asian",
    "indian",
    "black",
    "white",
    "middle eastern",
    "latino hispanic",
]
GENDER_CATEGORIES = ["man", "woman"]


def find_images(images_dir: Path):
    """Devuelve todas las imagenes en real/ y fake/ (la demografia no depende de la clase)."""
    paths = []
    for true_label in ("real", "fake"):
        subdir = images_dir / true_label
        if not subdir.is_dir():
            continue
        for path in sorted(subdir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTS:
                paths.append(str(path))
    return paths


def run_mock(paths, min_age: int):
    """Genera demografia aleatoria (siempre adultos) sin cargar ningun modelo."""
    rng = random.Random(7)
    out = []
    for path in paths:
        age = rng.randint(min_age, 70)
        out.append((path, age, rng.choice(GENDER_CATEGORIES), rng.choice(RACE_CATEGORIES)))
    return out


def run_real(paths, min_age: int):
    """Corre deepface.analyze imagen por imagen y filtra menores."""
    from deepface import DeepFace

    out = []
    skipped_minor = 0
    for path in paths:
        try:
            results = DeepFace.analyze(
                img_path=path,
                actions=["age", "gender", "race"],
                enforce_detection=False,
            )
        except Exception as exc:
            print(f"[warn] deepface fallo en {path}: {exc}")
            continue
        # deepface devuelve lista (un dict por rostro); usamos el primer rostro.
        face = results[0] if isinstance(results, list) else results
        age = int(face.get("age", 0))
        if age < min_age:
            skipped_minor += 1
            continue
        gender = str(face.get("dominant_gender", "")).strip().lower()
        race = str(face.get("dominant_race", "")).strip().lower()
        out.append((path, age, gender, race))
    if skipped_minor:
        print(f"[info] {skipped_minor} imagenes descartadas por edad < {min_age}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Etiquetar demografia -> data/labels.csv")
    parser.add_argument("--images", default="data/images",
                        help="Carpeta con subcarpetas real/ y fake/.")
    parser.add_argument("--out", default="data/labels.csv", help="CSV de salida.")
    parser.add_argument("--min_age", type=int, default=20,
                        help="Edad minima; se descartan rostros estimados por debajo.")
    parser.add_argument("--mock", action="store_true",
                        help="No carga deepface; genera demografia aleatoria.")
    args = parser.parse_args()

    images_dir = Path(args.images)
    paths = find_images(images_dir)
    if not paths:
        raise SystemExit(
            f"No se encontraron imagenes en {images_dir}/real y {images_dir}/fake. "
            "Coloca imagenes o usa --mock con archivos de prueba."
        )

    print(f"[info] {len(paths)} imagenes encontradas en {images_dir}")
    if args.mock:
        print("[info] modo --mock: demografia aleatoria, sin cargar deepface")
        rows = run_mock(paths, args.min_age)
    else:
        print("[info] cargando deepface (FairFace) ...")
        rows = run_real(paths, args.min_age)

    df = pd.DataFrame(rows, columns=["path", "age", "gender", "race"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[ok] escrito {out_path} ({len(df)} filas, solo adultos)")


if __name__ == "__main__":
    main()
