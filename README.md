# Titanic — Predicción de supervivencia

Contenido:
- `titanic.ipynb` — notebook con todo el análisis (EDA, preparación, modelos, evaluación, interpretación). Ya ejecutado, con salidas.
- `Informe_Titanic.docx` — documento con la solución y la justificación de cada paso.
- `data/` — `train.csv` y `test.csv` de https://www.kaggle.com/c/titanic/data
- `figs/` — gráficos generados por el notebook.
- `output/` — tablas de resultados y `submission.csv` para Kaggle.
- `build_notebook.py` / `build_informe.py` — scripts que generan el notebook y el informe.

## Cómo reproducir
```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace titanic.ipynb
python build_informe.py
```
Semilla fija (`SEED = 42`): los resultados son idénticos en cada ejecución.

## App web interactiva ("laboratorio")
En `app/` hay una aplicación Streamlit para experimentar con variables, modelos e hiperparámetros, simular
pasajeros y generar el envío a Kaggle. Reproduce los resultados del notebook (accuracy CV 0,841, hold-out 0,827).
```
pip install -r app/requirements.txt
streamlit run app/app.py
```
Detalles, publicación en Streamlit Community Cloud y cómo embeberla con `<iframe>`: [app/README.md](app/README.md).
