from values_xwyalpha import y
from y_predictiva import y_hat

e = []

# for i in range(3):
#     e.append(values.y[i] - yp.y_hat[i])

e.append(y[0] - y_hat[0]) # fila 1
e.append(y[1] - y_hat[1]) # fila 2
e.append(y[2] - y_hat[2]) # fila 3

#print("error e =", e)
