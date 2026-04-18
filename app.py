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
        
        # Seleccionamos las columnas que nos interesan (X + y)
        columnas_analisis = columnas_x + [columna_y]
        matriz_corr = df[columnas_analisis].corr()
        
        # --- 🛠️ SOLUCIÓN 1: Definimos un tamaño fijo (6 de ancho, 4 de alto) ---
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        
        # --- 🛠️ SOLUCIÓN 2: Le decimos a Streamlit que NO lo estire ---
        st.pyplot(fig, use_container_width=False)

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
            st.write("### 📉 Comparativa: Real vs Predicción")
            
            # --- 🛠️ SOLUCIÓN 3: Creamos columnas para centrar el gráfico ---
            col_izq, col_centro, col_der = st.columns([1, 2, 1])
            
            with col_centro:
                # 1. Crear el lienzo del gráfico con tamaño controlado
                fig_dispersion, ax_dispersion = plt.subplots(figsize=(7, 5))
                
                # 2. Dibujar los puntos (Real vs Predicción)
                ax_dispersion.scatter(y, predicciones, alpha=0.6, color='blue', label='Datos (Real vs Predicho)')
                
                # 3. Dibujar una línea roja punteada que representa la "predicción perfecta"
                limite_min = min(y.min(), predicciones.min())
                limite_max = max(y.max(), predicciones.max())
                ax_dispersion.plot([limite_min, limite_max], [limite_min, limite_max], color='red', linestyle='--', label='Línea de Predicción Perfecta')
                
                # 4. Etiquetas
                ax_dispersion.set_xlabel("Valores Reales")
                ax_dispersion.set_ylabel("Valores Predichos")
                ax_dispersion.legend() 
                
                # 5. Mostrar el gráfico en la interfaz de Streamlit
                st.pyplot(fig_dispersion)