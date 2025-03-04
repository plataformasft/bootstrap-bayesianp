import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import t  # Usamos scipy para el cálculo del intervalo t

st.title("Análisis Bootstrap y Bayesiano (Versión Simplificada)")

st.write("Sube un CSV con las columnas: **Tanque**, **Atributo** y **Dieta**")
uploaded_file = st.file_uploader("Cargar CSV", type=["csv"])

if uploaded_file is not None:
    # Cargar y mostrar los datos
    df = pd.read_csv(uploaded_file)
    st.subheader("Datos cargados")
    st.dataframe(df)
    
    # Verificar que las columnas existan
    if not set(["Tanque", "Atributo", "Dieta"]).issubset(df.columns):
        st.error("El archivo debe contener las columnas: Tanque, Atributo y Dieta")
    else:
        # Convertir la columna Atributo a numérico (por si viene como texto)
        df["Atributo"] = pd.to_numeric(df["Atributo"], errors="coerce")
        df = df.dropna(subset=["Atributo"])
        
        # Lista de grupos según la columna Dieta
        diets = df["Dieta"].unique()
        
        st.header("Análisis Bootstrap por grupo (Dieta)")
        n_boot = st.slider("Número de muestras bootstrap", min_value=100, max_value=5000, value=1000, step=100)
        
        for diet in diets:
            data = df[df["Dieta"] == diet]["Atributo"].values
            boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True))
                          for _ in range(n_boot)]
            mean_boot = np.mean(boot_means)
            ci_lower = np.percentile(boot_means, 2.5)
            ci_upper = np.percentile(boot_means, 97.5)
            st.write(f"**Dieta: {diet}** – Media Bootstrap: {mean_boot:.2f} | Intervalo 95%: [{ci_lower:.2f}, {ci_upper:.2f}]")
            
            # Gráfico de la distribución de las medias Bootstrap usando Altair
            boot_df = pd.DataFrame({"Medias Bootstrap": boot_means})
            chart = alt.Chart(boot_df).mark_bar().encode(
                alt.X("Medias Bootstrap:Q", bin=alt.Bin(maxbins=50)),
                y='count()'
            ).properties(
                title=f"Distribución Bootstrap para Dieta: {diet}"
            )
            st.altair_chart(chart, use_container_width=True)
        
        st.header("Análisis Bayesiano Simplificado (Aproximación)")
        st.write("Usando la media y el error estándar de cada grupo, calculamos un intervalo basado en la distribución *t* (aproximación de un intervalo de credibilidad con priors no informativos).")
        
        for diet in diets:
            data = df[df["Dieta"] == diet]["Atributo"].values
            n = len(data)
            mean_val = np.mean(data)
            std_val = np.std(data, ddof=1)
            se = std_val / np.sqrt(n)
            dfree = n - 1
            t_crit = t.ppf(0.975, dfree)
            ci_lower = mean_val - t_crit * se
            ci_upper = mean_val + t_crit * se
            st.write(f"**Dieta: {diet}** – Media: {mean_val:.2f} | Intervalo aproximado 95%: [{ci_lower:.2f}, {ci_upper:.2f}]")
        
        st.write("Nota: Esta aproximación asume un modelo normal y un prior no informativo, por lo que el intervalo se asemeja al intervalo de credibilidad bayesiano en esos casos.")

