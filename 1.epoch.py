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

batch_size = 3
batch_count = len(horas_estudio) // batch_size + (1 if len(horas_estudio) % batch_size > 0 else 0)
batch_segemnts = [horas_estudio[i*batch_size:(i+1)*batch_size] for i in range(batch_count)]

y_batch_size = batch_size
y_batch_count = len(resultado_real) // y_batch_size + (1 if len(resultado_real) % y_batch_size > 0 else 0)
y_batch_segments = [resultado_real[i*y_batch_size:(i+1)*y_batch_size] for i in range(y_batch_count)]

for epoch in range(batch_count):
    print(f"Epoch #{epoch + 1}")
    for batch, batch_y in zip(batch_segemnts[epoch], y_batch_segments[epoch]):
        x = batch
        y_real = batch_y
        
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