"""
Clasificación binaria con TensorFlow/Keras.
Predice si un estudiante aprueba o no según sus horas de estudio,
usando una sola neurona con activación sigmoid entrenada con
entropía cruzada y descenso de gradiente estocástico.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

"""
Dataset: 25 muestras donde X son horas de estudio (decimal)
y Y es el resultado (0 = reprobó, 1 = aprobó).
"""

from manage_data import fn
csvFn = fn.CSV()
dataset = csvFn.read("dataset.csv", headers=True)
TranformTime = fn.TextToTime("%H:%M:%S")
dataset = [[round(TranformTime.timetext_to_hours(x), 4), int(y)] for x, y in dataset]

X = np.array([x[0] for x in dataset], dtype=float)
y = np.array([x[1] for x in dataset], dtype=int)

"""
Configuración de mini-batches: divide los 25 datos en 5 lotes
de 5 muestras cada uno. Cada época procesa los 5 lotes secuencialmente.
"""
total_datos = len(X)
lotes_deseados = 5
tamano_lote = total_datos // lotes_deseados

"""
Modelo: una sola capa Dense con 1 neurona y activación sigmoid.
Calcula z = x·w + b y luego aplica σ(z) para obtener una probabilidad.
"""
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1], activation='sigmoid')
])

"""
Compilación: usa binary_crossentropy como función de pérdida (estándar
para clasificación binaria) y SGD como optimizador con learning rate 0.01.
"""
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

"""
Entrenamiento: 10 épocas con 5 steps por época (un step por batch).
En cada step el modelo ajusta pesos y bias según el gradiente del error.
"""
print("Iniciando entrenamiento...\n")
model.fit(
    X, y, 
    epochs=10, 
    batch_size=tamano_lote, 
    verbose=1
)

"""
Resultados: extrae los pesos aprendidos y hace una predicción de prueba
para ver la probabilidad de aprobar con 1.8 horas de estudio.
"""
pesos, bias = model.layers[0].get_weights()
print(f"\nPesos finales -> w: {pesos[0][0]:.4f}, b: {bias[0]:.4f}")

horas_prueba = np.array([2])
prediccion = model.predict(horas_prueba, verbose=0)
print(f"Probabilidad de aprobar con {horas_prueba[0]} horas: {(round(prediccion[0][0], 2) * 100):.2f}%")

print("\nClasificación (umbral 50%):", "Aprobar" if prediccion[0][0] >= 0.5 else "Reprobar")
