
from values_xwyalpha import X, w

#!Esto corresponde a cuanto espero que aprenda realmente el pokémon despues de los combates
y_hat = []

y_hat.append(X[0][0] * w[0] + X[0][1] * w[1])  # Fila 1
y_hat.append(X[1][0] * w[0] + X[1][1] * w[1])  # Fila 2
y_hat.append(X[2][0] * w[0] + X[2][1] * w[1])  # Fila 3

# for i in range(3):          # 3 filas
#     valor = 0
#     for j in range(2):      # 2 columnas
#         valor = valor + values.X[i][j] * values.w[j]
#     y_hat.append(valor)

#print("y_hat =", y_hat)