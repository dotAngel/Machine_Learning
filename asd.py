from matplotlib import pyplot as plt

# Leer datos desde un archivo CSV
data = None
with open("./data.csv", "r") as file:
    # Leer y organizar en listas los datos
    data = [line.split(",") for line in file.read().splitlines()]

# quitar headers
data = data[1:]

# Filtar datos, limpieza, parseo de tipos y redondeo
# columna altura = entradas; columna peso = weights
entradas = [float(e[0]) / 100 for e in data]
weights = [float(e[1]) / 10 for e in data]


# print("Entradas (altura en metros):", entradas)
# print("Pesos (en kg):", weights)

# Numero de entradas
n = len(entradas)

# Calcular sumatorias
sx = sum(entradas)
sy = sum(weights)
sxy = sum(xi * yi for xi, yi in zip(entradas, weights))
sx2 = sum(xi**2 for xi in entradas)

# Calcular pendiente (m)
m = (n * sxy - sx * sy) / (n * sx2 - sx**2)

# Calcular intercepto (b)
b = (sy - m * sx) / n

# Imprimir resultados
print("Pendiente m =", m)
print("Intercepto b =", b)
print(format(f"Ecuacion de la recta: y = {m} * x {'+' if b >=0 else '-'} {abs(b)}"))

# nuevas predicciones
pred = [
    1.10,
    1.15,
    1.20
]

y_pred = [round(m * xi + b, 2) for xi in pred]

# Imprimir nueva predicción
for i in range(len(pred)):
    print(format(f"Peso estimado para altura {pred[i]}m = {y_pred[i]} kg"))
    
plt.scatter(entradas, weights, color="blue", label="Datos originales")
plt.plot(pred, y_pred, color="red", label="Recta de regresión")
plt.xlabel("Altura (m)")
plt.ylabel("Peso (kg)")
plt.title("Regresión Lineal: Peso vs Altura")
plt.legend()
plt.grid()
plt.show()