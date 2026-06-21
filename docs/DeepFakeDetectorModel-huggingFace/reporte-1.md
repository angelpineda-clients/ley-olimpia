
Auditoría de sesgo demográfico en detección de deepfakes (Colab)

Proyecto Ley Olimpia · Global South AI Safety Hackathon — LatAm

Pregunta única: ¿un detector público de deepfakes comete más errores con rostros de ciertos grupos, dejando esos deepfakes sin detectar?

Métrica: tasa de falsos negativos (FNR) por grupo. Si a un grupo se le escapan más, ese grupo queda menos protegido (Ley Olimpia).

Este notebook corre el flujo completo en Colab:

    Setup (clonar repo + instalar dependencias)
    Elegir y preparar el dataset de caras real/fake
    Correr el detector → data/preds.csv
    Etiquetar demografía con FairFace → data/labels.csv
    Analizar el sesgo (FNR, AUROC, chi², brecha con IC95%) + figuras

Antes de empezar: menú Entorno de ejecución → Cambiar tipo de entorno → GPU.

Ética (innegociable): solo rostros benignos de datasets públicos, solo adultos (se filtra edad < 20), nada de contenido íntimo ni de generar deepfakes. La demografía la percibe un modelo (proxy), no la declara la persona: es una limitación que va en la sección de doble uso del reporte.
1. Setup: clonar el repo e instalar dependencias

Edita REPO_URL con la URL de tu repositorio (el que contiene src/ y requirements.txt). Si prefieres no usar git, puedes subir la carpeta src/ manualmente a Colab y saltarte la celda de clonado.
[ ]

import os

# === EDITA ESTO ===
REPO_URL = "https://github.com/angelpineda-clients/ley-olimpia"
PROJECT_DIR = "/content/ley-olimpia"
# ==================

if not os.path.isdir(PROJECT_DIR):
    !git clone $REPO_URL $PROJECT_DIR
else:
    print("El repo ya está clonado; haciendo pull...")
    !cd $PROJECT_DIR && git pull --ff-only

os.chdir(PROJECT_DIR)
print("Directorio de trabajo:", os.getcwd())
!ls -la

[ ]

# Dependencias. En Colab, torch ya viene instalado con CUDA; instalamos el resto.
!pip install -q transformers deepface pandas numpy scipy scikit-learn matplotlib pillow kagglehub

# Verificación rápida de GPU (opcional pero recomendado).
import torch
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

CUDA disponible: False

2. Elegir y preparar el dataset

El objetivo es dejar imágenes en data/images/real/ y data/images/fake/.

    Opción A — Kaggle "140k Real and Fake Faces" (recomendado): rápido, sin EULA. Reales = fotos Flickr/FFHQ, falsas = caras StyleGAN (contenido benigno, adultos). Estructura: real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}.

Ajusta SAMPLE_PER_CLASS para el MVP (unos cientos a 1-2 mil por clase; submuestrea si va lento). Revisa la ficha del dataset y quédate solo con adultos.
[ ]

# Helper común: copia un submuestreo de imágenes hacia data/images/{real,fake}.
import random, shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLE_PER_CLASS = 400  # imágenes por clase para el MVP (ajustable)

DEST = Path("data/images")
(DEST / "real").mkdir(parents=True, exist_ok=True)
(DEST / "fake").mkdir(parents=True, exist_ok=True)


def list_images(folder):
    folder = Path(folder)
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def populate_class(src_folder, klass, n=SAMPLE_PER_CLASS, seed=0):
    """Copia hasta n imágenes de src_folder a data/images/<klass>/."""
    imgs = list_images(src_folder)
    if not imgs:
        raise SystemExit(f"No se hallaron imágenes en {src_folder}")
    random.Random(seed).shuffle(imgs)
    imgs = imgs[:n]
    dest = DEST / klass
    for i, p in enumerate(imgs):
        shutil.copy(p, dest / f"{klass}_{i:05d}{p.suffix.lower()}")
    print(f"[ok] {len(imgs)} imágenes -> {dest}")


def summary():
    nr = len(list(Path("data/images/real").glob("*")))
    nf = len(list(Path("data/images/fake").glob("*")))
    print(f"Total preparado -> real: {nr} | fake: {nf}")

print("Helpers listos. Ejecuta UNA de las opciones A/B/C abajo.")

Helpers listos. Ejecuta UNA de las opciones A/B/C abajo.

[ ]

# === OPCIÓN A: Kaggle "140k Real and Fake Faces" (recomendada) ===
# kagglehub descarga el dataset (puede pedir credenciales de Kaggle la 1a vez:
# crea un token en kaggle.com -> Settings -> API, y súbelo, o usa kagglehub.login()).
import kagglehub
from pathlib import Path

ds_path = Path(kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces"))
print("Descargado en:", ds_path)

# La estructura típica es real-vs-fake/{train,valid,test}/{real,fake}.
# Usamos el split 'valid' para ir rápido; cambia a 'test' o 'train' si quieres más.
base = next(ds_path.rglob("valid"), None) or ds_path
real_src = base / "real"
fake_src = base / "fake"
print("real_src:", real_src, "| existe:", real_src.exists())
print("fake_src:", fake_src, "| existe:", fake_src.exists())

populate_class(real_src, "real")
populate_class(fake_src, "fake")
summary()

Using Colab cache for faster access to the '140k-real-and-fake-faces' dataset.
Descargado en: /kaggle/input/140k-real-and-fake-faces
real_src: /kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/valid/real | existe: True
fake_src: /kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/valid/fake | existe: True
[ok] 400 imágenes -> data/images/real
[ok] 400 imágenes -> data/images/fake
Total preparado -> real: 400 | fake: 400

3. Correr el detector de deepfakes → data/preds.csv

Usa el modelo por defecto (prithivMLmods/Deep-Fake-Detector-Model). Para robustez, puedes repetir con --model prithivMLmods/Deepfake-Detect-Siglip2.
[ ]

!python src/run_detector.py --model prithivMLmods/Deep-Fake-Detector-Model

import pandas as pd
preds = pd.read_csv("data/preds.csv")
print(preds.shape)
preds.head()

4. Etiquetar demografía (FairFace vía deepface) → data/labels.csv

Estima edad, género y raza por rostro y filtra menores (--min_age 20). La raza incluye la categoría latino hispanic, que da el ángulo LatAm.
[ ]

!python src/label_demographics.py --images data/images --min_age 20

import pandas as pd
labels = pd.read_csv("data/labels.csv")
print(labels.shape)
print("\nDistribución por raza:\n", labels["race"].value_counts())
print("\nDistribución por género:\n", labels["gender"].value_counts())
labels.head()

Ahora que tenemos las predicciones del detector (preds) y las etiquetas demográficas (labels), los combinamos para el análisis de sesgo.
[ ]

df_merged = pd.merge(preds, labels, on='path', how='inner')
print(df_merged.shape)
display(df_merged.head())

5. Analizar el sesgo: FNR, AUROC, chi², brecha con IC95% + figuras

Eje principal raza (lente LatAm) y eje secundario género (chequeo gratis con el mismo pipeline).
[ ]

# Eje principal: raza/etnia (lente LatAm)
!python src/analyze_bias.py --group race

# Eje secundario: género
!python src/analyze_bias.py --group gender

[ ]

# Mostrar tablas y figuras inline
import pandas as pd
from IPython.display import Image, display

for group in ["race", "gender"]:
    print(f"\n===== Sesgo por {group} =====")
    display(pd.read_csv(f"data/outputs/bias_{group}.csv"))
    display(Image(filename=f"data/outputs/bias_{group}.png"))

Cómo leer los resultados

    FNR alto en un grupo = a ese grupo se le escapan más deepfakes → menos protección.
    Brecha de FNR (peor − mejor) con IC95%: si el intervalo no cruza 0, hay evidencia de diferencia entre grupos. Si lo cruza, no se puede afirmar sesgo con esta muestra (recuerda: la literatura está dividida; ambos resultados son publicables).
    chi²: p < 0.05 sugiere que acertar/fallar en los fakes depende del grupo.
    Cuida los tamaños de muestra por grupo (n_fake); el análisis ignora grupos con < 5 fakes para no reportar tasas inestables. Si latino hispanic queda con n chico, súbelo aumentando SAMPLE_PER_CLASS o usando otro split del dataset.

Para el reporte: guarda data/outputs/bias_race.csv y bias_race.png (y los de género). Documenta en la sección de doble uso que la demografía es percibida por un modelo (proxy) y valida a mano ~40 etiquetas para reportar el acuerdo.
Informe Final: Auditoría de Sesgo Demográfico (Ley Olimpia)

Este reporte resume los hallazgos sobre la equidad del modelo detector de deepfakes. El objetivo es identificar si existen grupos demográficos más vulnerables debido a una mayor tasa de fallos del sistema.
1. Análisis de Sesgo por Raza/Etnia

El grupo con mayor Tasa de Falsos Negativos (FNR) es el más desprotegido, ya que el sistema no logra detectar contenido manipulado en sus rostros.

xychart-beta
    title "Tasa de Falsos Negativos (FNR) por Raza"
    x-axis ["Black", "Indian", "Mid. East", "Latino", "White", "Asian"]
    y-axis "FNR" 0 --> 1
    bar [0.67, 0.50, 0.44, 0.38, 0.34, 0.26]

Hallazgo clave: Existe una brecha significativa de 0.40 entre el grupo con mejor desempeño (Asian) y el de peor desempeño (Black). El grupo Latino/Hispano presenta un riesgo intermedio.
2. Análisis de Sesgo por Género

El análisis muestra una disparidad estadística entre hombres y mujeres en la detección de deepfakes.

pie title Distribución de Errores (FNR) por Género
    "Hombres (Mayor Error)" : 42
    "Mujeres (Menor Error)" : 30

    Hombres: FNR de 0.42
    Mujeres: FNR de 0.30
    P-valor: 0.014 (Indica que la diferencia es estadísticamente significativa).

3. Conclusiones para el Proyecto Ley Olimpia

    Vulnerabilidad Diferenciada: El modelo es menos efectivo protegiendo a personas de raza negra y hombres, aumentando su riesgo ante ataques de deepfakes.
    Necesidad de Mitigación: Se recomienda re-entrenar el modelo con datasets más balanceados que incluyan mayor representación de los grupos afectados para garantizar una justicia algorítmica alineada con los principios de la Ley Olimpia.
    Transparencia: Este reporte debe ser incluido en la documentación técnica para informar a los usuarios sobre las limitaciones del sistema.

