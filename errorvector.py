import values_xwyalpha as values
import y_predictiva as yp

e = []

# for i in range(3):
#     e.append(values.y[i] - yp.y_hat[i])

e.append(values.y[0] - yp.y_hat[0])
e.append(values.y[1] - yp.y_hat[1])
e.append(values.y[2] - yp.y_hat[2])

#print("error e =", e)
