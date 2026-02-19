def calcular_pendiente (n, sumx, sumy, sumxy, sumx2):
    # m = (noEntradas * sumatoria(xy) - sumatoria(x) * sumatoria(y)) / (noEntradas * sumatoria(x^2) - (sumatoria(x))^2)
    a = n * sumxy - sumx * sumy
    b = n * sumx2 - sumx**2
    n = a / b
    return n
    