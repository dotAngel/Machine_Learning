# (y - y^)**2 
from values_xwyalpha import y
from y_predictiva import y_hat
from nueva_prediccion import y_hat2

loss_0 = (((y[0] - y_hat[0]) ** 2) + ((y[1] - y_hat[1]) ** 2) + ((y[2] - y_hat[2]) ** 2)) / 3
loss_1 = (((y[0] - y_hat2[0]) ** 2) + ((y[1] - y_hat2[1]) ** 2) + ((y[2] - y_hat2[2]) ** 2)) / 3
