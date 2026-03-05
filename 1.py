horas_estudio = [1,2,3,4,5,6]
resultado_real = [0,0,0,1,1,1]

for h in horas_estudio:
    if h >= 4:
        prediccion = 1
    else:
        prediccion = 0
    
    print("Horas:", h, " -> Prediccion:", prediccion)

peso = 0
bias = 0
alpha = 0.1

for i in range(len(horas_estudio)):
    x = horas_estudio[i]
    y_real = resultado_real[i]
    
    z = peso * x + bias
    
    if z >= 0:
        y_pred = 1
    else:
        y_pred = 0
        
    error = y_real - y_pred
    
    peso = peso + alpha * error * x
    bias = bias + alpha * error
    
print("Peso:", peso)
print("Bias:", bias)

def pred (x):
    return peso * x + bias

print(pred(3))