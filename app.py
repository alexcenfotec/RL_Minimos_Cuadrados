import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ModeloRegresionMultiple import ModeloRegresionMultiple

# Título de la aplicación
st.set_page_config(page_title="Predictor de Regresión Lineal", layout="wide")
st.title("🤖 Predictor de Regresión Lineal con Mínimos Cuadrados")

# 1. CARGA DE ARCHIVOS
archivo_subido = st.file_uploader("Sube tu dataset (CSV)", type=["csv"])

if archivo_subido is not None:
    df = pd.read_csv(archivo_subido)
    st.write("### Vista previa de los datos")
    st.dataframe(df.head())

    # 2. CONFIGURACIÓN DEL MODELO (Barra lateral)
    st.sidebar.header("Configuración del Modelo")
    columna_y = st.sidebar.selectbox("Selecciona la columna objetivo (y)", df.columns)
    columnas_x = st.sidebar.multiselect("Selecciona las columnas de características (X)", df.columns)

    # Visualización: Mapa de calor
    if st.checkbox("Mostrar mapa de calor de correlación"):
        st.write("### 🌡️ Matriz de Correlación")
        columnas_analisis = columnas_x + [columna_y]
        matriz_corr = df[columnas_analisis].corr()
        
        fig, ax = plt.subplots()
        sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # 3. ENTRENAMIENTO DEL MODELO
    if columnas_x and columna_y:
        # Extraemos los valores para el modelo
        X = df[columnas_x].values
        y = df[columna_y].values

        if st.button("Entrenar Modelo"):
            # Instanciamos y entrenamos usando la clase refactorizada
            modelo = ModeloRegresionMultiple(X, y)
            modelo.ajustar()
            
            # Realizamos predicciones
            predicciones = modelo.predecir(X)
            
            # --- LLAMADA A LOS MÉTODOS DE LA CLASE ---
            rmse = modelo.evaluar_rmse(y, predicciones)
            r2 = modelo.evaluar_r2(y, predicciones)
            
            # Mostramos resultados en columnas
            st.write("### Resultados del Entrenamiento")
            c1, c2 = st.columns(2)
            c1.metric("RMSE (Error promedio)", f"{rmse:.4f}")
            c2.metric("Coeficiente de Determinación R²", f"{r2:.4f}")

            # 4. GRÁFICO COMPARATIVO
            st.write("### 📉 Comparativa: Valores Reales vs Predichos")
            fig_disp, ax_disp = plt.subplots(figsize=(10, 6))
            
            # Puntos de dispersión
            ax_disp.scatter(y, predicciones, alpha=0.5, color='royalblue', label='Datos del modelo')
            
            # Línea de referencia ideal (Predicción perfecta)
            limite = [min(y.min(), predicciones.min()), max(y.max(), predicciones.max())]
            ax_disp.plot(limite, limite, color='red', linestyle='--', linewidth=2, label='Predicción Ideal')
            
            ax_disp.set_xlabel("Valores Reales (y)")
            ax_disp.set_ylabel("Valores Predichos (ŷ)")
            ax_disp.legend()
            st.pyplot(fig_disp)