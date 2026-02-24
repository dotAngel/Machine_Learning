import pandas as pd
import os
from calcular_m import calcular_pendiente
from calcular_b import calcular_intercepto

# Leer datos desde CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'data_pandas.csv')
df = pd.read_csv(csv_path)
x = df['x'].tolist()
y = df['y'].tolist()
predecir = 15

# Calcular m y b
n = len(x)
sum_x = sum(x)
sum_y = sum(y)
sum_xy = sum(xi * yi for xi, yi in zip(x, y))
sum_x2 = sum(xi**2 for xi in x)

m = calcular_pendiente(n, sum_x, sum_y, sum_xy, sum_x2)
b = calcular_intercepto(n, sum_x, sum_y, m)

# Función para predecir
def pred(x_val):
    return m * x_val + b

def sf(op):
    if op:
        print("Valor de m =", m)
        print("Valor de b =", b)
    else:
        print(f"Ecuacion de la recta: y = {m} X  {f'- {abs(b)}' if b < 0 else f'+ {b}'}")

print(f"Valor para la prediccion de x = {predecir} es y = {pred(predecir)}")
sf(0)
