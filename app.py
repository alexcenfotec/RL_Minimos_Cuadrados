import streamlit as st
import pandas as pd
import numpy as np  # Lo necesitamos para crear la matriz de los nuevos datos
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
    st.dataframe(df.head(10))  # Mostramos las primeras 10 filas para una mejor visualización

    # 2. CONFIGURACIÓN DEL MODELO (Barra lateral)
    st.sidebar.header("Configuración del Modelo")
    columna_y = st.sidebar.selectbox("Selecciona la columna objetivo (y)", df.columns)
    columnas_x = st.sidebar.multiselect("Selecciona las columnas de características (X)", df.columns)

    # Visualización: Mapa de calor
    if st.checkbox("Mostrar mapa de calor de correlación"):
        col_izq, col_centro, col_der = st.columns([1, 2, 1])
        with col_centro:
            st.write("### 🌡️ Matriz de Correlación")
            columnas_analisis = columnas_x + [columna_y]
            matriz_corr = df[columnas_analisis].corr()
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig, use_container_width=False)

    # 3. ENTRENAMIENTO DEL MODELO
    if columnas_x and columna_y:
        X = df[columnas_x].values
        y = df[columna_y].values

        if st.button("Entrenar Modelo"):
            modelo = ModeloRegresionMultiple(X, y)
            modelo.ajustar()
            
            # --- 🛠️ Guardamos el modelo en la memoria de Streamlit ---
            st.session_state['modelo_entrenado'] = modelo
            
            predicciones = modelo.predecir(X)
            rmse = modelo.evaluar_rmse(y, predicciones)
            r2 = modelo.evaluar_r2(y, predicciones)
            
            col_izq_res, col_centro_res, col_der_res = st.columns([1, 2, 1])
            with col_centro_res:
                st.write("### Resultados del Entrenamiento")
                c1, c2 = st.columns(2)
                c1.metric("RMSE (Error promedio)", f"{rmse:.4f}")
                c2.metric("Coeficiente de Determinación R²", f"{r2:.4f}")
                
                st.write("---")
                st.write("### 📉 Comparativa: Real vs Predicción")
                
                fig_dispersion, ax_dispersion = plt.subplots(figsize=(7, 5))
                ax_dispersion.scatter(y, predicciones, alpha=0.6, color='blue', label='Datos (Real vs Predicho)')
                limite_min = min(y.min(), predicciones.min())
                limite_max = max(y.max(), predicciones.max())
                ax_dispersion.plot([limite_min, limite_max], [limite_min, limite_max], color='red', linestyle='--', label='Línea de Predicción Perfecta')
                
                ax_dispersion.set_xlabel("Valores Reales")
                ax_dispersion.set_ylabel("Valores Predichos")
                ax_dispersion.legend() 
                st.pyplot(fig_dispersion, use_container_width=False)

    # =====================================================================
    # 4. PREDICCIÓN INTERACTIVA CON NUEVOS DATOS
    # =====================================================================
    
    # Solo mostramos esta sección si el modelo ya fue entrenado y guardado en memoria
    if 'modelo_entrenado' in st.session_state:
        st.write("---")
        st.write("### 🔮 Realizar Nuevas Predicciones")
        st.markdown("Ingresa nuevos valores para predecir el resultado basándonos en el modelo entrenado:")
        
        # Creamos columnas dinámicamente según la cantidad de variables X elegidas
        columnas_inputs = st.columns(len(columnas_x))
        valores_nuevos = []
        
        # Generamos una cajita de texto (number_input) para cada variable característica
        for i, col_name in enumerate(columnas_x):
            with columnas_inputs[i]:
                valor = st.number_input(f"Ingresa el valor para '{col_name}'", value=0.0, step=0.5)
                valores_nuevos.append(valor)
                
        # Botón para ejecutar la predicción
        if st.button("Calcular Predicción"):
            # 1. Convertimos la lista de valores en una matriz de 1 fila: [[val1, val2...]]
            X_nuevo = np.array([valores_nuevos])
            
            # 2. Extraemos el modelo de la memoria y llamamos al método predecir
            modelo_guardado = st.session_state['modelo_entrenado']
            resultado = modelo_guardado.predecir(X_nuevo)
            
            # 3. Mostramos el resultado con un cuadro de éxito (verde)
            st.success(f"🌟 El valor predicho para **{columna_y}** es: **{resultado[0]:.2f}**")