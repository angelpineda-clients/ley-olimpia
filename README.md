# Sesgo demográfico en detección de deepfakes (pivote empírico)

Versión simple y con datos del proyecto Olimpia, como pidieron los mentores. En vez de
auditar políticas (varias preguntas, "otra fase"), una sola pregunta empírica:

**¿Un detector público de deepfakes comete más errores con rostros de ciertos grupos
(piel/etnia o género), dejando esos deepfakes sin detectar?**

Si un detector se le escapan más los deepfakes de un grupo, las personas de ese grupo
quedan menos protegidas frente a contenido íntimo no consentido sintético. Ahí está el
puente con Ley Olimpia: la eliminación depende de esa detección.

## Métrica
Tasa de falsos negativos (FNR) por grupo = entre las imágenes que SON fake, qué fracción
el detector marcó como real. FNR alto = a ese grupo se le escapan más deepfakes. También:
accuracy y AUROC por grupo, prueba chi² y brecha de FNR con IC95% por bootstrap.

## El método (ya validado en la literatura)
Etiquetar la demografía de cada rostro con un modelo (FairFace), correr el detector, y
medir su error por grupo. Es el mismo enfoque del apéndice de TalkingHeadBench (2025).
Ojo: la literatura está dividida (USC 2021 y UB 2024 hallan sesgo racial; un estudio de
2026 no halla sesgo sistemático). Eso vuelve la pregunta interesante: replicar con un
detector público actual y lente LatAm es una contribución legítima, salga como salga.

## Datos (esto desbloquea su pendiente)
No necesitan un dataset pre-etiquetado por demografía: etiquetan ustedes con FairFace.
- Rostros reales+fake, descargables: cualquier dataset abierto de "real vs fake faces"
  (p. ej. en Kaggle/Hugging Face; los fake suelen ser caras StyleGAN, contenido benigno).
  Revisen la ficha del dataset y quédense solo con adultos.
- Estándar de oro (más lento, requieren solicitud/EULA): FaceForensics++, DFDC, y
  Casual Conversations de Meta (sujetos con consentimiento, etiquetados por edad, género
  y tono de piel; diseñado justo para evaluar equidad en deepfakes).
- Etiquetas demográficas: las pone `deepface` (envuelve FairFace), que da raza
  (incluye "latino hispanic"), género y edad.

## Modelos (verificados, se cargan con transformers)
Detector de deepfakes (image-classification):
- prithivMLmods/Deep-Fake-Detector-Model (ViT)
- prithivMLmods/Deepfake-Detect-Siglip2
- prithivMLmods/deepfake-detector-model-v1
Usen 1 para el MVP; un 2do da robustez.

## Cómputo
Es inferencia de imágenes: Google Colab (GPU gratis) o local. NO es Groq. Groq sirve para
modelos de texto; aquí corren modelos de visión con `transformers`/`deepface`.

## Flujo
```bash
pip install transformers torch deepface pandas matplotlib

# 1) Imágenes en data/images/real/*.jpg y data/images/fake/*.jpg
# 2) Detector -> data/preds.csv
python src/run_detector.py --model prithivMLmods/Deep-Fake-Detector-Model
# 3) Demografía (adultos) -> data/labels.csv
python src/label_demographics.py --images data/images --min_age 20
# 4) Sesgo por raza o género -> tabla, prueba, figura
python src/analyze_bias.py --group race
python src/analyze_bias.py --group gender
```
Para probar el armazón sin descargar nada: agreguen `--mock` a los scripts 2 y 3.

## Alcance MVP (24h, 3 personas)
- 1 detector, 1 dataset (unos cientos a 1–2 mil imágenes; submuestrear si va lento).
- Eje primario: raza/etnia con lente LatAm (categoría "latino hispanic" de FairFace).
  Eje secundario, gratis con el mismo pipeline: género (lo que sugirieron los mentores).
- Validar a mano ~40 etiquetas demográficas y reportar el acuerdo.

## Ética (innegociable)
Solo rostros benignos de datasets públicos. Adultos (filtrar age<20). Nada de contenido
íntimo ni de generar deepfakes. La demografía es percibida por un modelo, no declarada:
es un proxy y una limitación; va en la sección de doble uso del reporte.

## Cómo conecta con el track
Sigue siendo Gobernanza de IA (auditoría y rendición de cuentas) + Responsible AI
(equidad). Ahora con evidencia empírica, que es lo que pedían los mentores.
# ley-olimpia
