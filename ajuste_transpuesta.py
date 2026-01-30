from values_xwyalpha import X
from errorvector import e

grad = []

grad.append((X[0][0] * e[0] + X[1][0] * e[1] + X[2][0] * e[2]))
grad.append((X[0][1] * e[0] + X[1][1] * e[1] + X[2][1] * e[2]))

# for j in range(2):          # columnas
#     suma = 0
#     for i in range(3):      # filas
#         suma = suma + X[i][j] * e[i]
#     grad[j] = suma

# print("X^T e =", grad)
