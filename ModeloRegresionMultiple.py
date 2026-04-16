import numpy as np

class ModeloRegresionMultiple:
    def __init__(self, X, y):
        """
        Constructor de la clase.
        Prepara los datos para el cálculo de la regresión.
        """
        # Obtenemos el número de filas (muestras) del dataset
        n_muestras = X.shape[0]
        
        # 🏗️ Construcción de la Matriz de Diseño:
        # Para incluir el 'intercepto' (beta 0) en la multiplicación de matrices,
        # agregamos una columna llena de unos al inicio de nuestras variables X.
        columna_unos = np.ones((n_muestras, 1))
        self.X = np.column_stack((columna_unos, X))
        
        # Guardamos la variable objetivo (y)
        self.y = y
        
        # Aquí se guardarán los resultados (Betas) una vez entrenado el modelo
        self.coeficientes = None

    def ajustar(self):
        """
        Entrena el modelo usando la Ecuación Normal de Mínimos Cuadrados.
        Fórmula: beta = (X^T * X)^-1 * X^T * y
        """
        # 1. Calculamos la transpuesta de X multiplicada por X (X^T * X)
        xt_x = self.X.T @ self.X
        
        # 2. Calculamos la inversa de ese resultado
        inversa_xt_x = np.linalg.inv(xt_x)
        
        # 3. Multiplicamos por la transpuesta de X y luego por el vector y
        # El resultado son los coeficientes óptimos (Betas)
        self.coeficientes = inversa_xt_x @ (self.X.T @ self.y)

    def predecir(self, X_nuevo):
        """
        Toma nuevos datos y predice el valor de 'y' usando los coeficientes calculados.
        """
        n_muestras_nuevas = X_nuevo.shape[0]
        
        # También debemos agregar la columna de unos a los datos nuevos para que la matriz coincida
        columna_unos = np.ones((n_muestras_nuevas, 1))
        X_nuevo_transformado = np.column_stack((columna_unos, X_nuevo))
        
        # Realizamos el producto punto entre los datos y los coeficientes (Y = X * Beta)
        return X_nuevo_transformado @ self.coeficientes

    def evaluar_mse(self, y_real, y_pred):
        """
        Calcula el Error Cuadrático Medio (MSE).
        Mide el promedio de los errores al cuadrado.
        """
        # (Valor Real - Valor Predicho)^2
        errores_cuadrados = (y_real - y_pred) ** 2
        # Retornamos el promedio de esos errores
        return errores_cuadrados.mean()

    def evaluar_rmse(self, y_real, y_pred):
        """
        Calcula la Raíz del Error Cuadrático Medio (RMSE).
        Es la métrica en las mismas unidades que la variable 'y'.
        """
        # Aplicamos la raíz cuadrada al MSE calculado arriba
        return np.sqrt(self.evaluar_mse(y_real, y_pred))

    def evaluar_r2(self, y_real, y_pred):
        """
        Calcula el Coeficiente de Determinación (R²).
        Indica qué porcentaje de la variación de los datos explica el modelo.
        """
        # 1. Suma de Cuadrados Totales (Variación total de los datos reales)
        ss_tot = np.sum((y_real - y_real.mean()) ** 2)
        
        # 2. Suma de Cuadrados de los Residuos (Variación que el modelo NO explicó)
        ss_res = np.sum((y_real - y_pred) ** 2)
        
        # 3. Fórmula final: 1 - (Error / Variación Total)
        return 1 - (ss_res / ss_tot)