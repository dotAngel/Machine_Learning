import random
from manage_data import fn
from activation import fn as activation_fn


Vector = fn.Vector
sigmoid = activation_fn.sigmoid


class Logistic_Regresion:
    """
    Modelo de regresión logística con descenso de gradiente por mini-batches.
    Sirve tanto para clasificación binaria como para ajustar cualquier frontera
    de decisión no lineal, dependiendo de la función de activación elegida.
    """
    
    def __init__(self, alpha=0.01, epochs=1, verbose=False, batch_size=None, batch_count=1, fn="sigmoid", weights=None, bias=None, verbose_nwb=False, verbose_epoch=False):
        self.alpha = alpha
        self.epochs = epochs
        self.weights = weights
        self.bias = bias
        self.verbose = verbose
        self.verbose_nwb = verbose_nwb
        self.verbose_epoch = verbose_epoch
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.fn = fn
        
        
    def fit(self, x, y, **kwargs):
        """
        Entrena el modelo ajustando pesos y bias mediante descenso de gradiente.
        Divide los datos en mini-batches, calcula el forward pass aplicando la
        función de activación, obtiene los gradientes del error y actualiza los
        parámetros en cada iteración. Es el corazón del aprendizaje.
        """
        
        if self.batch_count == 1 and self.batch_size is None:
            self.batch_size = len(x)
        elif self.batch_size is None:
            self.batch_size = len(x) // self.batch_count
            
        if not isinstance(x[0], (list, tuple)):
            x = [[val] for val in x]
            
        n_samples = len(x)
        n_features = len(x[0])
        
        if self.weights is None:
            w_raw = [random.uniform(-1, 1) for _ in range(n_features)]
            self.weights = Vector(w_raw)
        else:
            self.weights = Vector(self.weights)
        
        if self.bias is None:
            self.bias = random.uniform(-1, 1)
            
        if self.verbose:
            print(f"Initial weights: {self.weights.data}")
            print(f"Initial bias: {self.bias}")

        for epoch in range(self.epochs):
            for batch in range(self.batch_count):
                
                if self.verbose:
                    print(f"\nEpoch {epoch+1}/{self.epochs}, Batch {batch+1}/{self.batch_count}")
                
                start = batch * self.batch_size
                end = min(start + self.batch_size, n_samples)
                
                x_batch = Vector(x[start:end])
                y_batch = y[start:end]

                z = x_batch @ self.weights
                z = [val + self.bias for val in z]
                
                if not self.fn == "none":
                    y_hat = [getattr(activation_fn, self.fn)(val) for val in z]
                else:
                    y_hat = z

                dw = [0.0] * n_features
                db = 0.0
                batch_len = len(y_batch)

                for i in range(batch_len):
                    error = y_hat[i] - y_batch[i] 
                    for j in range(n_features):
                        dw[j] += error * x[start+i][j]
                    db += error

                new_w = [self.weights.data[j] - (self.alpha * dw[j] / batch_len) for j in range(n_features)]
                self.weights = Vector(new_w)
                self.bias -= self.alpha * db / batch_len

            if self.verbose or self.verbose_epoch:
                print(f"Epoch {epoch+1}/{self.epochs} completada.")
                
                if self.verbose_nwb:
                    print(f"new weights: {self.weights.data}")
                    print(f"new bias: {self.bias}")
        
        if self.verbose:
            print(f"\nFinal weights: {self.weights.data}")
            print(f"Final bias: {self.bias}")
            
            
    def predict(self, x, fn='sigmoid'):
        """
        Genera predicciones continuas para nuevas entradas. Aplica la combinación
        lineal (X·W + b) seguida de la función de activación. Devuelve valores
        crudos (probabilidades si es sigmoid, valores reales si es "none").
        """        
        if not isinstance(x, (list, tuple)):
            x = [[x]]
            
        if not isinstance(x[0], (list, tuple)):
            x = [[val] for val in x]
        
        x_vec = Vector(x)
        z = x_vec @ self.weights
        z = [val + self.bias for val in z]
        if not fn == "none":
            y_hat = [getattr(activation_fn, fn)(val) for val in z]
        else:
            y_hat = z
        return y_hat
    
    
    def evaluate(self, x, y, fn='sigmoid'):
        """
        Mide la precisión del modelo comparando sus predicciones con los valores
        reales. Usa un umbral fijo de 0.5 para convertir probabilidades en
        clases y devuelve el porcentaje de aciertos. Útil para saber si el
        modelo realmente aprendió o solo memoriza ruido.
        """
        y_hat = self.predict(x, fn=fn)
        y_pred = [1 if val >= 0.5 else 0 for val in y_hat]
        accuracy = sum(1 for yp, yt in zip(y_pred, y) if yp == yt) / len(y)
        return accuracy
    
    
    def classify(self, x, threshold=0.5, fn='sigmoid'):
        """
        Convierte las predicciones en etiquetas binarias (0 o 1) según un umbral
        configurable. Es lo que usas cuando ya no te interesa la probabilidad
        sino la decisión final: "¿aprueba o no aprueba?".
        """
        y_hat = self.predict(x, fn=fn)
        return [1 if val >= threshold else 0 for val in y_hat]