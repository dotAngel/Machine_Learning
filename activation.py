import math

class fn:
    
    # Range: (0, 1)
    # Uso: Clasificación binaria (capa de salida), probabilidades, regresión logística.
    @staticmethod
    def sigmoid (x):
        return 1 / (1 + math.exp(-x))
    
    # Range: [0, infinite)
    # Uso: Capas ocultas en redes profundas (CNN, MLP). Es la opción por defecto en la mayoría de arquitecturas.
    @staticmethod
    def relu (x):
        return max(0, x)
    
    # Range: (-1, 1)
    # Uso: Capas ocultas cuando se necesitan salidas centradas en 0 (ej. RNNs, LSTMs).
    @staticmethod
    def tanh (x):
        return math.tanh(x)
    
    # Range: (-infinite, infinite)
    # Uso: Capas ocultas donde ReLU causa "neuronas muertas" (gradiente 0 para x < 0).
    @staticmethod
    def leaky_relu (x, alpha=0.01):
        return max(alpha * x, x)
    
    # Range: (0, 1)
    # Uso: Clasificación multiclase (capa de salida). Devuelve distribución de probabilidades.
    @staticmethod
    def softmax (x):
        exp_x = [math.exp(i) for i in x]
        sum_exp_x = sum(exp_x)
        return [i / sum_exp_x for i in exp_x]
    
    # Range: (-alpha, infinite)
    # Uso: Alternativa a ReLU en capas ocultas. Suaviza valores negativos, mejora convergencia.
    @staticmethod
    def elu(x, alpha=1.0):
        return x if x > 0 else alpha * (math.exp(x) - 1)

    # Range: [~-0.28, infinite)
    # Uso: Capas ocultas en redes profundas. Supera a ReLU en modelos muy grandes (ej. EfficientNet).
    @staticmethod
    def swish(x):
        return x / (1 + math.exp(-x))

    # Range: (0, infinite)
    # Uso: Cuando se necesita una versión suave de ReLU, o como salida que debe ser positiva (ej. varianzas).
    @staticmethod
    def softplus(x):
        return math.log(1 + math.exp(x))

    # Range: [~-0.17, infinite)
    # Uso: Capas ocultas en Transformers y modelos de NLP (ej. BERT, GPT).
    @staticmethod
    def gelu(x):
        # Usamos la aproximación matemática para no saturar el CPU
        return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))