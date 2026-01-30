from values_xwyalpha import X, w1
from ajuste_pesos import weight_adjust

y_hat2 = []

# for i in range(3):
#     valor = 0
#     for j in range(2):
#         valor = valor + X[i][j] * w[j]
#     y_hat2.append(valor)

weight_adjust()

y_hat2.append(X[0][0] * w1[0] + X[0][1] * w1[1])  # Fila 1
y_hat2.append(X[1][0] * w1[0] + X[1][1] * w1[1])  # Fila 2
y_hat2.append(X[2][0] * w1[0] + X[2][1] * w1[1])  # Fila 3

# print("y_hat2 =", y_hat2)
