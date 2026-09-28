"""Cada página se ejecuta sin excepciones (streamlit.testing.v1.AppTest)."""
import pytest
from streamlit.testing.v1 import AppTest

from conftest import APP

PAGINAS = ["pages/1_Explorar_datos.py", "pages/2_Ingenieria_de_variables.py", "pages/3_Entrenar_modelos.py",
           "pages/4_Interpretacion.py", "pages/5_Simulador.py", "pages/6_Envio_Kaggle.py"]


def nueva_app():
    at = AppTest.from_file(str(APP / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


def ir(at, pagina):
    at.switch_page(pagina).run()
    assert not at.exception, at.exception
    return at


def boton(at, texto):
    return next(b for b in at.button if texto in b.label)


def test_inicio():
    at = nueva_app()
    assert "Laboratorio Titanic" in at.title[0].value


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_corre(pagina):
    ir(nueva_app(), pagina)


def test_entrenar_registra_historial_y_reproduce_notebook():
    at = ir(nueva_app(), "pages/3_Entrenar_modelos.py")
    boton(at, "Entrenar").click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["historial"]) == 1
    assert at.session_state["historial"][0]["Acc CV"] == pytest.approx(0.841, abs=0.01)
    assert any(m.label == "Acc CV" for m in at.metric)


@pytest.mark.parametrize("modelo", ["Regresión logística", "KNN", "SVM", "Árbol de decisión", "Gradient Boosting"])
def test_entrenar_cada_modelo_e_interpretar(modelo):
    at = ir(nueva_app(), "pages/3_Entrenar_modelos.py")
    at.selectbox(key="w_p_modelo").set_value(modelo).run()
    at.slider(key="w_p_folds").set_value(3).run()
    boton(at, "Entrenar").click().run()
    assert not at.exception, at.exception
    ir(at, "pages/4_Interpretacion.py")
    ir(at, "pages/5_Simulador.py")


def test_curva_de_validacion():
    at = ir(nueva_app(), "pages/3_Entrenar_modelos.py")
    at.selectbox(key="w_p_modelo").set_value("Árbol de decisión").run()
    at.slider(key="w_p_folds").set_value(3).run()
    boton(at, "Calcular curva").click().run()
    assert not at.exception, at.exception


def test_busqueda_en_grilla_y_aplicar():
    at = ir(nueva_app(), "pages/3_Entrenar_modelos.py")
    at.selectbox(key="w_p_modelo").set_value("KNN").run()
    at.slider(key="w_p_folds").set_value(3).run()
    boton(at, "Lanzar búsqueda").click().run()
    assert not at.exception, at.exception
    boton(at, "Aplicar estos hiperparámetros").click().run()
    assert not at.exception, at.exception


def test_sin_variables_muestra_error():
    at = ir(nueva_app(), "pages/2_Ingenieria_de_variables.py")
    for cb in at.checkbox:
        if cb.key and cb.key.startswith("w_p_var_"):
            cb.uncheck()
    at.run()
    assert not at.exception
    assert any("Desactivaste todas" in e.value for e in at.error)
    ir(at, "pages/3_Entrenar_modelos.py")
    assert any("No hay variables activas" in e.value for e in at.error)


def test_configuracion_persiste_entre_paginas():
    at = ir(nueva_app(), "pages/2_Ingenieria_de_variables.py")
    at.selectbox(key="w_p_imputacion").set_value("knn").run()
    ir(at, "pages/1_Explorar_datos.py")
    ir(at, "pages/2_Ingenieria_de_variables.py")
    assert at.selectbox(key="w_p_imputacion").value == "knn"


def test_envio_kaggle_valida_y_descarga():
    at = ir(nueva_app(), "pages/6_Envio_Kaggle.py")
    boton(at, "891").click().run()
    assert not at.exception, at.exception
    assert any("Formato validado" in s.value for s in at.success)
