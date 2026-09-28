Quiero construir una aplicación web interactiva ("laboratorio") sobre el problema del Titanic de Kaggle, para embeber en una página web y que permita experimentar cambiando parámetros y ver el efecto en los modelos.

## Contexto (lo que ya existe en esta carpeta)
- `data/train.csv`, `data/test.csv`, `data/gender_submission.csv`: datos oficiales de Kaggle.
- `titanic.ipynb` y `build_notebook.py`: análisis completo ya hecho. El código del notebook está escrito como texto dentro de `build_notebook.py`, así que **no se modifica**: la app tiene su propia copia en `app/core/` y un test garantiza que, con los valores por defecto, produce exactamente lo mismo que el notebook. Reutilizá su lógica, en especial:
  - la clase `TitanicFeatures` (transformer de scikit-learn que crea Title, AgeImp, AgeMissing, IsChild, FamilySize, FamilyGroup, TicketGroup, HasCabin, Deck, LogFare, e imputa Embarked/Fare aprendiendo solo del set de entrenamiento),
  - la función `armar_pipeline(modelo, escalar)` (TitanicFeatures + ColumnTransformer con StandardScaler/OneHotEncoder + modelo),
  - el esquema de evaluación: hold-out estratificado 20% (`train_test_split(..., stratify=y, random_state=SEED)`) + `StratifiedKFold(10, shuffle=True, random_state=SEED)` aplicado **solo sobre el 80% de entrenamiento**, SEED = 42.
- Resultados de referencia en `output/*.csv`:
  - `tuning.csv`: Random Forest ajustado (max_depth=4, max_features=0.5, min_samples_leaf=3, 300 árboles) → Acc train 0,849 / **Acc CV 0,841** ± 0,037. "Acc train" es `mean_train_score` de la CV (`return_train_score=True`), no el modelo reentrenado.
  - `holdout.csv`: Random Forest 0,827 accuracy y 0,847 ROC-AUC en hold-out; base "mujeres viven" **0,777 en el hold-out** (≈0,787 sobre el 80% de entrenamiento).
- En la raíz ya existen `README.md` y `requirements.txt` del notebook: **no pisarlos**.

## Stack
- Python + **Streamlit** (se puede publicar gratis en Streamlit Community Cloud y embeber en cualquier web con un `<iframe>` usando `?embed=true`).
- pandas, numpy, scikit-learn, plotly (gráficos interactivos), matplotlib (solo para dibujar el árbol con `plot_tree`).
- Estructura del proyecto:
  ```
  app/
    app.py              # página de inicio (Streamlit multipágina)
    pages/              # una página por sección (1_Explorar_datos.py, 2_..., etc.)
    core/data.py        # carga de datos con rutas relativas a __file__ (funciona local y en Cloud)
    core/features.py    # TitanicFeatures parametrizable, variables disponibles, armar_pipeline
    core/models.py      # catálogo de modelos, hiperparámetros editables, escalado por modelo, grillas chicas
    core/evaluation.py  # CV, hold-out, métricas, línea base, curva de validación, búsqueda en grilla, permutación
    core/state.py       # configuración compartida entre páginas en st.session_state
    tests/              # pytest + streamlit.testing.v1.AppTest
    requirements.txt    # versiones fijadas (Streamlit Cloud lo busca junto al archivo principal)
    README.md           # cómo correr local, cómo publicar y cómo embeber
  .streamlit/config.toml  # tema
  ```
- Todo en **español** (textos, etiquetas, ayudas).

## `TitanicFeatures` parametrizable
El constructor recibe (con defaults que reproducen el notebook exactamente):
- `edad_nino=13` (umbral de IsChild), `cortes_familia=(1, 4)` (Solo ≤1 / Chica ≤4 / Grande),
- `imputacion_edad="titulo"` ∈ {`"titulo"` mediana por título, `"mediana"` global, `"media"`, `"knn"` KNNImputer con Pclass, SibSp, Parch, LogFare y Sex codificado, escalados, k=5},
- `variables=None` (lista de variables de salida; `None` = todas las del notebook: NUM + CAT).

`Title` se calcula siempre internamente (lo necesita la imputación por título) aunque el usuario no lo use como variable del modelo. `armar_pipeline(modelo, escalar, **params_features)` arma el `ColumnTransformer` solo con las variables numéricas y categóricas activas.
Variables originales seleccionables: Pclass, Sex, SibSp, Parch, Embarked (el modelo no usa `Age`/`Fare` crudas sino `AgeImp`/`LogFare`).

## Estado compartido entre páginas
En `st.session_state`: configuración de variables (página 2), configuración de modelo y evaluación (página 3), último pipeline entrenado e historial. Si una página necesita un modelo y todavía no se entrenó ninguno, se usa el Random Forest por defecto (entrenado y cacheado) y se muestra un aviso.

## Páginas / funcionalidades

1. **Explorar datos**
   - Filtros en la barra lateral (sexo, clase, puerto, rango de edad, tamaño de familia). Los filtros **solo afectan la exploración**, nunca los datos de entrenamiento.
   - Gráficos interactivos de tasa de supervivencia por variable elegida por el usuario, histogramas de Age/Fare con color por Survived, tabla cruzada Sex × Pclass, mapa de faltantes.
   - Cada gráfico con una frase explicando qué muestra.

2. **Ingeniería de variables**
   - Checkboxes para activar/desactivar cada variable derivada (Title, FamilySize, FamilyGroup, IsChild, HasCabin, Deck, LogFare, TicketGroup, AgeMissing, AgeImp) y cada variable original (Pclass, Sex, SibSp, Parch, Embarked).
   - Selector de estrategia de imputación de edad: mediana global / mediana por título / media / KNNImputer.
   - Umbral de "niño" configurable (slider, por defecto 13) y cortes de FamilyGroup configurables.
   - Vista previa de las primeras filas transformadas y conteo de nulos resultante.
   - Botón "Restaurar valores del notebook".

3. **Entrenar y comparar modelos (el "playground")**
   - Selector de modelo: Regresión logística, KNN, SVM, Árbol de decisión, Random Forest, Gradient Boosting. El escalado se decide solo: sí para LR/KNN/SVM, no para árboles y ensambles.
   - Controles de hiperparámetros según el modelo: C y penalty (solo l1/l2, con `solver="liblinear"`); k y weights; C, kernel y gamma (`probability=True`); max_depth, min_samples_leaf, criterion; n_estimators (máx. 500), max_depth, max_features, min_samples_leaf; learning_rate, n_estimators (máx. 500), max_depth, subsample.
   - Controles de evaluación: cantidad de folds (3–10), % de hold-out (10–40%), semilla.
   - Botón "Entrenar". Mostrar: accuracy de entrenamiento (media de train en la CV, como el notebook) vs CV (con desvío) vs hold-out, **brecha de sobreajuste** (train − CV) resaltada con color si supera 0,05, ROC-AUC, F1, matriz de confusión, curva ROC, y comparación contra la base "mujeres viven" calculada sobre el mismo hold-out.
   - **Historial de experimentos** en `st.session_state`: tabla con cada corrida (modelo, parámetros, variables usadas, métricas), posibilidad de marcar favoritas, borrar y descargar como CSV. Gráfico comparando las corridas.
   - Curva de validación: elegir un hiperparámetro y graficar accuracy train vs CV al variarlo (para ver sobreajuste/subajuste).
   - Búsqueda en grilla chica (≤ 24 combinaciones) implementada como bucle sobre `ParameterGrid` + `cross_validate` para poder actualizar `st.progress` (GridSearchCV no expone progreso).

4. **Interpretación**
   - Importancia por permutación sobre el hold-out, pasando el pipeline completo (es decir, sobre las **columnas originales**, como el notebook), con explicación de por qué `Name` aparece importante.
   - Coeficientes / odds ratio si es regresión logística (nombres vía `get_feature_names_out`); `feature_importances_` si es un árbol o ensamble.
   - Si es árbol de decisión, dibujarlo con `plot_tree` limitando la profundidad visible (slider).

5. **Simulador "¿Hubiera sobrevivido?"**
   - Formulario con sexo, edad (con opción "desconocida"), clase, tarifa, puerto, hermanos/cónyuge, padres/hijos, título, cabina sí/no (+ cubierta si sí) y cantidad de personas con el mismo pasaje.
   - Se arma una fila compatible con el pipeline: `Name` sintético `"Pasajero, {Título}. X"`, `Ticket` sintético que produzca el TicketGroup elegido, `Cabin` = `"{Cubierta}1"` o nulo. Avisar si el título es incoherente con sexo/edad (p. ej. Master mujer).
   - Muestra la probabilidad de sobrevivir según el último modelo entrenado, con un indicador visual, y cómo cambia al modificar una variable a la vez (gráfico de probabilidad vs edad y vs tarifa manteniendo el resto fijo; barras por clase y por sexo).

6. **Envío a Kaggle**
   - Entrenar el modelo actual con los 891 registros, predecir `test.csv` y descargar `submission.csv` con el formato exacto. Validar antes de habilitar la descarga: 418 filas, columnas `PassengerId,Survived`, `PassengerId` idénticos y en el mismo orden que `test.csv`, `Survived` entero 0/1, sin nulos.

## Requisitos técnicos
- Sin fuga de información: todo el preprocesamiento dentro del Pipeline, ajustado solo con datos de entrenamiento de cada fold.
- `@st.cache_data` para la carga de datos y resultados; `@st.cache_resource` para pipelines entrenados. La configuración se pasa como JSON ordenado (string) para que sea hasheable.
- Pensado para Streamlit Community Cloud (≈1 GB RAM, CPU limitada): límites en sliders, grillas chicas, `n_jobs=-1` solo donde ayuda.
- Semilla fija por defecto para que los resultados sean reproducibles; mostrar la configuración completa de cada corrida.
- Manejar errores con mensajes claros (por ejemplo, si el usuario desactiva todas las variables).
- Diseño limpio: barra lateral para controles, métricas principales con `st.metric`, pestañas dentro de cada página. Modo claro/oscuro: dejar que Streamlit lo maneje y usar `st.plotly_chart(..., theme="streamlit")`, sin colores de fondo fijos.
- `app/README.md` con:
  1. cómo correrla local (`pip install -r app/requirements.txt` y `streamlit run app/app.py`),
  2. cómo publicarla en Streamlit Community Cloud desde un repo de GitHub (subir también `data/`; archivo principal `app/app.py`),
  3. el snippet de `<iframe src="https://<mi-app>.streamlit.app/?embed=true" width="100%" height="900"></iframe>` para embeberla en una web.

## Cómo trabajar
- Prioridad: primero una versión mínima con las páginas 1, 3, 5 y 6 usando las variables del notebook; después la página 2, la 4, la curva de validación y la búsqueda en grilla.
- "Probar que cada página corra" significa: `pytest` con (a) un test que compare `TitanicFeatures()` por defecto contra la versión original del notebook, (b) un `AppTest` por página que la ejecute sin excepciones.
- Al terminar, verificá que con la configuración por defecto (Random Forest, max_depth=4, max_features=0.5, min_samples_leaf=3, 300 árboles, SEED=42, 10 folds sobre el 80%) se obtenga Acc CV ≈ 0,841 y hold-out ≈ 0,827, igual que en el notebook.
