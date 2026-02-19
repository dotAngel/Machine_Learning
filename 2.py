# Datos
x = [1.50, 1.60, 1.70]  # alturas
y = [50, 60, 65]        # pesos

def calc(altura_nueva):
    n = len(x)

    # Calcular sumatorias
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi**2 for xi in x)

    # Calcular pendiente (m)
    # m = (noEntradas * sumatoria(xy) - sumatoria(x) * sumatoria(y)) / (noEntradas * sumatoria(x^2) - (sumatoria(x))^2)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

    # Calcular intercepto (b)
    # b = (sumatoria(y) - m * sumatoria(x)) / noEntradas
    b = (sum_y - m * sum_x) / n

    print("Pendiente m =", m)
    print("Intercepto b =", b)

    # Predicción
    #nueva prediccion = (pendiente)(x_nueva) + bias; m = pendiente; b = bias
    # peso_predicho = m * altura_nueva + b
    peso_predicho = (m*altura_nueva+b)
        
    print("Peso estimado para altura", altura_nueva, "=", peso_predicho, "kg")
    
    x.append(altura_nueva)
    y.append(peso_predicho)
        
for i in range(3): 
    calc([1.65,1.67,1.69][i])
    
print(x,y)