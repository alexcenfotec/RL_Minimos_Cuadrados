import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ModeloRegresionMultiple import ModeloRegresionMultiple

# Título de la aplicación
st.title("🤖 Predictor de Regresión Lineal con Mínimos Cuadrados")

# 1. CARGA DE ARCHIVOS
archivo_subido = st.file_uploader("Sube tu dataset (CSV)", type=["csv"])

if archivo_subido is not None:
    df = pd.read_csv(archivo_subido)
    st.write("### Vista previa de los datos")
    st.dataframe(df.head())

    # 2. CONFIGURACIÓN DEL MODELO (En la barra lateral)
    st.sidebar.header("Configuración del Modelo")

    # Seleccionar la variable objetivo (y)
    columna_y = st.sidebar.selectbox("Selecciona la columna objetivo (y)", df.columns)

    # Seleccionar las variables independientes (X)
    columnas_x = st.sidebar.multiselect("Selecciona las columnas de características (X)", df.columns)

    # Mapa de calor de correlación
    if st.checkbox("Mostrar mapa de calor de correlación"):
        st.write("### 🌡️ Matriz de Correlación")
        
        # Seleccionamos las columnas que nos interesan (X + y)
        columnas_analisis = columnas_x + [columna_y]
        matriz_corr = df[columnas_analisis].corr()
        
        # Creamos la figura de Matplotlib
        fig, ax = plt.subplots()
        sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        
        # Lo mostramos en Streamlit
        st.pyplot(fig)

    # 3. EXTRACCIÓN DE DATOS
    if columnas_x and columna_y:
        # Creamos X con las columnas seleccionadas por el usuario
        X = df[columnas_x].values
        
        # Creamos y con la columna objetivo
        y = df[columna_y].values
        
        st.success("¡Datos preparados con éxito!")

        # 4. ENTRENAMIENTO DEL MODELO
        if st.button("Entrenar Modelo"):
            # Instanciar y ajustar usando la clase que creaste
            modelo = ModeloRegresionMultiple(X, y)
            modelo.ajustar()
            
            # Generar predicciones y evaluar
            predicciones = modelo.predecir(X)
            mse = modelo.evaluar_mse(y, predicciones)
            
            # --- NUEVO: Cálculo del RMSE usando numpy ---
            rmse = np.sqrt(mse) 
            
            r2 = modelo.evaluar_r2(y, predicciones)
            
            # Mostrar las métricas en la interfaz
            st.write("### Resultados del Entrenamiento")
            col1, col2 = st.columns(2) # Creamos dos columnas
            
            with col1:
                # --- NUEVO: Mostramos el RMSE en lugar del MSE ---
                st.metric(label="Raíz del Error Cuadrático Medio (RMSE)", value=round(rmse, 4))
            with col2:
                st.metric(label="Coeficiente R²", value=round(r2, 4))

            # 5. GRÁFICO COMPARATIVO
            st.write("### 📉 Comparativa: Real vs Predicción")
            
            # 1. Crear el lienzo del gráfico
            fig_dispersion, ax_dispersion = plt.subplots()
            
            # 2. Dibujar los puntos (Real vs Predicción) --- NUEVO: Agregamos label ---
            ax_dispersion.scatter(y, predicciones, alpha=0.6, color='blue', label='Datos (Real vs Predicho)')
            
            # 3. Dibujar una línea roja punteada que representa la "predicción perfecta" --- NUEVO: Agregamos label ---
            limite_min = min(y.min(), predicciones.min())
            limite_max = max(y.max(), predicciones.max())
            ax_dispersion.plot([limite_min, limite_max], [limite_min, limite_max], color='red', linestyle='--', label='Línea de Predicción Perfecta')
            
            # 4. Etiquetas
            ax_dispersion.set_xlabel("Valores Reales")
            ax_dispersion.set_ylabel("Valores Predichos")
            
            # --- NUEVO: Activamos la leyenda ---
            ax_dispersion.legend() 
            
            # 5. Mostrar el gráfico en la interfaz de Streamlit
            st.pyplot(fig_dispersion)