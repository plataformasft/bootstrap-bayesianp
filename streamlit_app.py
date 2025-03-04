import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

# Función para realizar el análisis Bootstrap
def bootstrap_analysis(data, n_resamples=1000):
    def statistic(x, axis):
        return np.mean(x, axis=axis)
    
    res = bootstrap((data,), statistic, n_resamples=n_resamples, vectorized=True)
    return res

# Configuración de la página
st.title("Análisis Bootstrap de Dietas")
st.write("Carga un archivo CSV con las columnas 'Tanque', 'Dieta' y 'Atributo' para realizar el análisis.")

# Seleccionar el delimitador
delimitador = st.radio("Selecciona el delimitador del archivo CSV", [",", ";", "|", "Tabulación"])

# Convertir "Tabulación" a "\t"
if delimitador == "Tabulación":
    delimitador = "\t"

# Cargar el archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo CSV con el delimitador seleccionado
    df = pd.read_csv(uploaded_file, delimiter=delimitador)
    
    # Mostrar las primeras filas del dataset
    st.write("Vista previa del dataset:")
    st.write(df.head())
    
    # Verificar que las columnas necesarias estén presentes
    if 'Tanque' not in df.columns or 'Dieta' not in df.columns or 'Atributo' not in df.columns:
        st.error("El archivo CSV debe contener las columnas 'Tanque', 'Dieta' y 'Atributo'.")
    else:
        # Seleccionar la dieta para analizar
        dietas = df['Dieta'].unique()
        dieta_seleccionada = st.selectbox("Selecciona una dieta para analizar", dietas)
        
        # Filtrar el dataset por la dieta seleccionada
        df_filtrado = df[df['Dieta'] == dieta_seleccionada]
        
        # Realizar el análisis Bootstrap
        st.write(f"Realizando análisis Bootstrap para la dieta: {dieta_seleccionada}")
        bootstrap_result = bootstrap_analysis(df_filtrado['Atributo'].values)
        
        # Mostrar los resultados
        st.write(f"Media del atributo: {np.mean(df_filtrado['Atributo'])}")
        st.write(f"Intervalo de confianza del 95%: {bootstrap_result.confidence_interval}")
        
        # Graficar los resultados
        fig, ax = plt.subplots()
        ax.hist(bootstrap_result.bootstrap_distribution, bins=30, edgecolor='k')
        ax.axvline(np.mean(df_filtrado['Atributo']), color='r', linestyle='dashed', linewidth=2)
        ax.axvline(bootstrap_result.confidence_interval.low, color='g', linestyle='dashed', linewidth=2)
        ax.axvline(bootstrap_result.confidence_interval.high, color='g', linestyle='dashed', linewidth=2)
        ax.set_title(f"Distribución Bootstrap para la dieta {dieta_seleccionada}")
        ax.set_xlabel("Valor del Atributo")
        ax.set_ylabel("Frecuencia")
        
        st.pyplot(fig)
else:
    st.write("Por favor, sube un archivo CSV para comenzar el análisis.")
