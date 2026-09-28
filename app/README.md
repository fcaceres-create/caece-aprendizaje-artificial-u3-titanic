# Laboratorio Titanic (app web)

**En vivo:** https://caece-aprendizaje-artificial-u3-titanic.streamlit.app

Aplicación Streamlit para experimentar con el problema Titanic de Kaggle: explorar los datos, elegir
variables, entrenar y comparar modelos, interpretarlos, simular pasajeros y generar el envío a Kaggle.
Reutiliza la lógica de `titanic.ipynb` (misma ingeniería de variables y mismo esquema de evaluación): con
la configuración por defecto da **accuracy CV 0,841 ± 0,037 y hold-out 0,827**, igual que el notebook, y el
`submission.csv` es idéntico al de `output/`.

## Páginas
| Página | Qué hace |
|---|---|
| Explorar datos | Filtros, supervivencia por variable, histogramas de edad y tarifa, sexo × clase, mapa de faltantes |
| Ingeniería de variables | Activar/desactivar variables, imputación de edad (título / mediana / media / KNN), umbral de niño, cortes de FamilyGroup |
| Entrenar modelos | 6 modelos con sus hiperparámetros, train vs CV vs hold-out, brecha de sobreajuste, ROC, matriz de confusión, historial, curva de validación, búsqueda en grilla |
| Interpretación | Importancia por permutación, coeficientes y odds ratio, `feature_importances_`, dibujo del árbol |
| ¿Hubiera sobrevivido? | Probabilidad para un pasajero inventado y cómo cambia al mover una variable |
| Envío a Kaggle | Reentrena con las 891 filas, valida el formato y descarga `submission.csv` |

## Estructura
```
.streamlit/config.toml     tema (claro/oscuro automático)
data/                      train.csv, test.csv, gender_submission.csv (los lee la app)
app/
  app.py                   punto de entrada y navegación
  pages/                   una página por sección
  core/features.py         TitanicFeatures (parametrizable) y armar_pipeline
  core/models.py           catálogo de modelos, hiperparámetros, curvas y grillas
  core/evaluation.py       hold-out + CV, métricas, curva de validación, grilla, permutación, validación del envío
  core/state.py            estado compartido entre páginas y funciones cacheadas
  core/data.py             carga de datos (rutas relativas, funciona local y en la nube)
  tests/                   pytest: equivalencia con el notebook + cada página con AppTest
  requirements.txt
```

## 1. Correrla en tu computadora
Desde la carpeta `TP_Titanic` (Python 3.12 recomendado):
```
pip install -r app/requirements.txt
streamlit run app/app.py
```
Se abre en http://localhost:8501.

Para correr los tests: `pip install pytest` y después `pytest app/tests`.

## 2. Publicarla en Streamlit Community Cloud (gratis)
1. Subí la carpeta `TP_Titanic` a un repositorio de GitHub. Tienen que estar `app/`, `data/` y `.streamlit/`
   (los archivos del notebook pueden quedar, no molestan).
2. Entrá a https://share.streamlit.io con tu cuenta de GitHub → **Create app** → **Deploy a public app from GitHub**.
3. Elegí el repositorio y la rama, y en **Main file path** poné `app/app.py`.
4. En **Advanced settings** elegí Python 3.12. Las dependencias se toman de `app/requirements.txt`.
5. Elegí la URL (por ejemplo `laboratorio-titanic.streamlit.app`) y presioná **Deploy**.

Cada `git push` al repositorio actualiza la app sola. Si nadie la usa por varios días, Streamlit la pone a
dormir; se despierta con un clic al visitarla.

## 3. Embeberla en una página web
Agregá `?embed=true` a la URL (oculta el menú y el pie de Streamlit):
```html
<iframe src="https://caece-aprendizaje-artificial-u3-titanic.streamlit.app/?embed=true" width="100%" height="900"
        style="border:none;" title="Laboratorio Titanic"></iframe>
```
Otras opciones: `?embed=true&embed_options=dark_theme` fuerza el modo oscuro y `embed_options=light_theme` el claro.

## Notas de diseño
- **Sin fuga de información:** toda la preparación (imputaciones, escalado, one-hot) está dentro del
  `Pipeline`, así que en cada fold se ajusta solo con los datos de entrenamiento de ese fold. El hold-out
  no interviene en ninguna decisión.
- **"Acc entrenamiento"** es la media de la accuracy de entrenamiento en cada fold (como en el notebook).
  La brecha train − CV se marca en rojo si supera 0,05.
- **Caché:** los modelos entrenados se guardan con `st.cache_resource` usando como clave la configuración
  completa en JSON. Repetir una configuración ya entrenada es instantáneo.
- **SVM:** las probabilidades salen de `CalibratedClassifierCV(SVC(), ensemble=False)`, porque
  `SVC(probability=True)` quedó obsoleto en scikit-learn 1.9.
- **Regresión logística:** usa `solver="liblinear"` con `l1_ratio` (0 = l2, 1 = l1), porque `penalty` quedó
  obsoleto en scikit-learn 1.8.
- **Simulador:** el Pipeline espera las columnas originales de Kaggle, así que el nombre, el ticket y la
  cabina se generan a partir del formulario para que produzcan el título, TicketGroup y Deck elegidos.
