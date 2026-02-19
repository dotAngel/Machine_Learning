def calcular_intercepto(n, sumx, sumy, m):
    # b = (sumatoria(y) - m * sumatoria(x)) / noEntradas
    a = sumy - m * sumx
    b = a / n
    return b
