import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Configuración de la página
st.title("ANOVA con Bootstrap para Atributo por Dieta")
st.write("Carga un archivo CSV con las columnas 'Dieta' (texto) y 'Atributo' (numérico) para realizar el análisis.")

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
    
    # Asegurarse de que la columna 'Dieta' sea tratada como texto (string)
    df['Dieta'] = df['Dieta'].astype(str)
    
    # Mostrar las primeras filas del dataset
    st.write("Vista previa del dataset:")
    st.write(df.head())
    
    # Verificar que las columnas necesarias estén presentes
    if 'Dieta' not in df.columns or 'Atributo' not in df.columns:
        st.error("El archivo CSV debe contener las columnas 'Dieta' (texto) y 'Atributo' (numérico).")
    else:
        # Número de iteraciones de bootstrap
        n_iter = st.slider("Número de iteraciones de Bootstrap", min_value=1000, max_value=10000, value=5000)
        
        # Seleccionar el nivel de confianza
        confianza = st.slider("Nivel de confianza (%)", min_value=80, max_value=99, value=95)
        alpha = (100 - confianza) / 100  # Convertir a nivel de significancia
        
        # Función para calcular la diferencia de medias entre los grupos
        def diff_medias(grupos):
            mean_diff = []  # Lista vacía para almacenar las diferencias
            n = len(grupos)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    mean_diff.append(np.mean(grupos[i]) - np.mean(grupos[j]))
            return mean_diff
        
        # Lista para almacenar las diferencias de medias obtenidas en el bootstrap
        bootstrap_diffs = []
        
        # Realizamos el bootstrap
        st.write(f"Realizando {n_iter} iteraciones de Bootstrap...")
        for _ in range(n_iter):
            # Re-muestreo con reemplazo para cada grupo (Dieta)
            bootstrap_grupos = [
                np.random.choice(df[df['Dieta'] == dieta]['Atributo'], size=len(df[df['Dieta'] == dieta]), replace=True)
                for dieta in df['Dieta'].unique()
            ]
            
            # Calcular la diferencia de medias para las muestras bootstrap
            bootstrap_diffs.append(diff_medias(bootstrap_grupos))
        
        # Convertir a matriz para facilitar cálculos
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calcular los intervalos de confianza empíricos
        ci_lower = np.percentile(bootstrap_diffs, (alpha / 2) * 100, axis=0)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100, axis=0)
        
        # Mostrar los resultados
        st.write(f"### Resultados del Bootstrap ({confianza}% de confianza)")
        st.write("Intervalos de confianza empíricos para las diferencias de medias entre las dietas:")
        
        # Obtener combinaciones de dietas
        dietas = df['Dieta'].unique()
        comb_dietas = list(combinations(dietas, 2))
        
        for i, (dieta1, dieta2) in enumerate(comb_dietas):
            st.write(f"Diferencia entre **{dieta1}** y **{dieta2}**: ({ci_lower[i]:.3f}, {ci_upper[i]:.3f})")
        
        # Graficar las diferencias de medias
        st.write("### Gráfico de las diferencias de medias")
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (dieta1, dieta2) in enumerate(comb_dietas):
            ax.hist(bootstrap_diffs[:, i], bins=30, alpha=0.5, label=f"{dieta1} vs {dieta2}")
            ax.axvline(ci_lower[i], color='r', linestyle='--', linewidth=1)
            ax.axvline(ci_upper[i], color='r', linestyle='--', linewidth=1)
        ax.set_xlabel("Diferencia de medias")
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribución de las diferencias de medias (Bootstrap, {confianza}% de confianza)")
        ax.legend()
        st.pyplot(fig)
else:
    st.write("Por favor, sube un archivo CSV para comenzar el análisis.")
