from values_xwyalpha import X, w1, w, alpha
from ajuste_transpuesta import grad

# for j in range(2):
#     w[j] = w[j] + alpha * grad[j]

def weight_adjust ():
    w1[0] = w[0] + alpha * grad[0]
    w1[1] = w[1] + alpha * grad[1]

# print("nuevos pesos w =", w)
