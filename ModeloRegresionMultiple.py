import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

class ModeloRegresionMultiple:
    def __init__(self, X, y):
        # Obtenemos la cantidad de filas (muestras)
        n_muestras = X.shape[0]
        
        # 🏗️ Construimos la matriz de diseño agregando la columna de unos al inicio
        columna_unos = np.ones((n_muestras, 1))
        self.X = np.column_stack((columna_unos, X))
        
        # Guardamos el vector de la variable dependiente
        self.y = y
        
# Método para ajustar el modelo de regresión múltiple utilizando mínimos cuadrados
    def ajustar(self):
        # Calculamos los coeficientes beta mediante mínimos cuadrados
        self.coeficientes = np.linalg.inv(self.X.T @ self.X) @ (self.X.T @ self.y)

# Método para predecir los valores utilizando los coeficientes ajustados
    def predecir(self, X_nuevo):
        # Obtenemos la cantidad de filas de los nuevos datos
        n_muestras_nuevas = X_nuevo.shape[0]
        
        # Creamos la columna de unos y la unimos a la matriz
        columna_unos = np.ones((n_muestras_nuevas, 1))
        X_nuevo_transformado = np.column_stack((columna_unos, X_nuevo))
        
        # Calculamos y retornamos las predicciones
        predicciones = X_nuevo_transformado @ self.coeficientes
        return predicciones 

# Método para evaluar el modelo utilizando el error cuadrático medio (MSE)
    def evaluar_mse(self, y_real, y_pred):
        mse_cuadrado = (y_real - y_pred) ** 2
        mse_final = mse_cuadrado.mean()
        return mse_final  


# Método para evaluar el modelo utilizando el coeficiente de determinación (R^2)
    def evaluar_r2(self, y_real, y_pred):
        # 1. Suma de Cuadrados Totales (SStot)
        ss_tot = np.sum((y_real - y_real.mean()) ** 2)
        
        # 2. Suma de Cuadrados de los Residuos (SSres)
        ss_res = np.sum((y_real - y_pred) ** 2)
        
        # 3. Cálculo final de R^2
        r2 = 1 - (ss_res / ss_tot)
        
        return r2 


# ==========================================
# 📂 SECCIÓN: CARGAR LOS DATOS
# ==========================================

# ------------------------------------------
# 1.1 Lectura de CSV
# ------------------------------------------

datosVino = pd.read_csv('winequality-red.csv')

# 2. Extraer la variable dependiente (y) y convertirla a NumPy
y_Vino = datosVino['quality'].values

# 3. Extraer las variables independientes (X) eliminando la columna objetivo y convirtiendo a NumPy
# (axis=1 le indica a pandas que queremos eliminar una columna, no una fila)
X_Vino = datosVino.drop('quality', axis=1).values 


# 1. Instanciar: Creamos el "objeto" pasándole nuestra materia prima
modelo_vino = ModeloRegresionMultiple(X_Vino, y_Vino)

# 2. Ajustar: Le pedimos al objeto que calcule la matemática interna
modelo_vino.ajustar()

# 3. Predecir: Le damos nuevos datos y nos devuelve las predicciones

prediccionesVino = modelo_vino.predecir(X_Vino) 

# 4. Comparativa visual de los primeros 5 casos
print("\n--- Comparativa: Real vs Predicción ---")
for i in range(5):
    real = y_Vino[i]
    predicho = prediccionesVino[i]
    diferencia = real - predicho
    print(f"Caso {i+1}: Real = {real} | Predicho = {predicho:.2f} | Diferencia = {diferencia:.2f}")

# 5. Evaluar: Comparamos las predicciones con los valores reales utilizando métricas de evaluación
mse_vino = modelo_vino.evaluar_mse(y_Vino, prediccionesVino)
r2_vino = modelo_vino.evaluar_r2(y_Vino, prediccionesVino)

print(f"MSE del Vino: {mse_vino}")
print(f"R² del Vino: {r2_vino}")