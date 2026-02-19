from data import x, y, predecir
from calcular_m import calcular_pendiente
from calcular_b import calcular_intercepto

n = len(x)
sum_x = sum(x)
sum_y = sum(y)
sum_xy = sum(xi * yi for xi, yi in zip(x, y))
sum_x2 = sum(xi**2 for xi in x)

m = calcular_pendiente(n, sum_x, sum_y, sum_xy, sum_x2)
b = calcular_intercepto(n, sum_x, sum_y, m)

def predecir(x):
    return m * x + b

def showform(op):
    if op:
        print("Valor de m =", m)
        print("Valor de b =", b)
    else:
        print(f"Ecuacion de la recta: y = {m} X  {f"- {b}" if b < 0 else f"+ {b}"}")