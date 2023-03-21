import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Crear DataFrame con los datos
data = pd.DataFrame({'Horas trabajadas': [35, 40, 25, 30, 45, 50, 20, 15, 10, 5],
                     'Ventas generadas': [1200, 1400, 900, 1000, 1600, 1800, 800, 700, 500, 200]})

# Preparar los datos para el análisis
X = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1].values.reshape(-1, 1)

# Dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Crear y entrenar el modelo de regresión lineal simple
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Visualizar los resultados del modelo
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Relación entre horas trabajadas y ventas generadas')
plt.xlabel('Horas trabajadas')
plt.ylabel('Ventas generadas')
plt.show()

# Realizar predicciones con el modelo
y_pred = regressor.predict(X_test)

# Evaluar la precisión del modelo
from sklearn.metrics import mean_squared_error, r2_score
print('Error cuadrático medio: ', mean_squared_error(y_test, y_pred))
print('Coeficiente de determinación: ', r2_score(y_test, y_pred))


