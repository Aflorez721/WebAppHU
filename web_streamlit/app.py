import streamlit as st
import os
import pandas as pd
from processingmodels import *

# Configuración inicial de Streamlit
st.set_page_config(
    page_title="Generador de Pronósticos",
    page_icon="📈",
    layout="centered"
)

# Título y descripción de la aplicación
st.title("Generador de Pronósticos 📈")
st.markdown("Sube un archivo Excel (.xlsx) para generar pronósticos y métricas.")

# Carpeta para almacenar archivos temporales
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Widget para subir el archivo Excel
uploaded_file = st.file_uploader("Selecciona un archivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Guardar el archivo temporalmente
    filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Archivo guardado: {uploaded_file.name}")

    # Ejecutar el script Python para generar pronósticos y métricas
    try:
        st.info("Procesando el archivo...")
        forecast_file, metrics_file = generate_forecast_and_metrics(filepath)

        # Mostrar enlaces para descargar los resultados
        st.subheader("Descargar Resultados")
        with open(forecast_file, "rb") as f:
            st.download_button(
                label="Descargar Pronósticos",
                data=f,
                file_name="pronosticos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with open(metrics_file, "rb") as f:
            st.download_button(
                label="Descargar Métricas",
                data=f,
                file_name="metricas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        st.success("Proceso completado exitosamente.")
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")