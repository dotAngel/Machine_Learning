
import values_xwyalpha as values

#!Esto corresponde a cuanto espero que aprenda realmente el pokémon despues de los combates
y_hat = []

y_hat.append(values.X[0][0] * values.w[0] + values.X[0][1] * values.w[1])  # Fila 1
y_hat.append(values.X[1][0] * values.w[0] + values.X[1][1] * values.w[1])  # Fila 2
y_hat.append(values.X[2][0] * values.w[0] + values.X[2][1] * values.w[1])  # Fila 3

# for i in range(3):          # 3 filas
#     valor = 0
#     for j in range(2):      # 2 columnas
#         valor = valor + values.X[i][j] * values.w[j]
#     y_hat.append(valor)

#print("y_hat =", y_hat)