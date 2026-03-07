class fn:
    """
    Módulo utilitario que agrupa herramientas para manipulación de datos.
    Contiene subclases para conversión de tiempo, lectura/procesamiento de CSVs
    y operaciones con vectores/matrices.
    """
    
    class TextToTime:
        """
        Convierte texto con formato de hora (HH:MM:SS) a unidades numéricas.
        Útil para transformar columnas de tiempo en features numéricos que un
        modelo pueda consumir directamente.
        """
        
        def __init__ (self, format="%H:%M:%S"):
            self.format = format
            
            if self.format != "%H:%M:%S":
                raise ValueError("Invalid format. Only '%H:%M:%S' is supported.")
        
        
        def time_from_text (self, time):
            """
            Reconstruye una cadena HH:MM:SS a partir de los segundos totales.
            Es la operación inversa: convierte a segundos y luego reformatea.
            """
            if not isinstance(time, str):
                raise ValueError("time must be a string in the format '%H:%M:%S'")
            if self.format != "%H:%M:%S":
                raise ValueError("Invalid format. Only '%H:%M:%S' is supported.")
            
            time = self.timetext_to_seconds(time)
            timetext_to_seconds = time % 60
            time = time // 60
            timetext_to_minutes = time % 60
            time = time // 60
            timetext_to_hours = time % 24
            
            return f"{timetext_to_hours:02d}:{timetext_to_minutes:02d}:{timetext_to_seconds:02d}"
        
        
        def timetext_to_seconds (self, time):
            """
            Convierte "HH:MM:SS" a su equivalente total en segundos.
            Es la unidad base desde la que se derivan minutos y horas.
            """
            if not isinstance(time, str):
                raise ValueError("time must be a string in the format '%H:%M:%S'")
            if self.format != "%H:%M:%S":
                raise ValueError("Invalid format. Only '%H:%M:%S' is supported.")
            
            time = time.split(":")
            time = [int(x) for x in time]
            return time[0] * 3600 + time[1] * 60 + time[2]
        
        
        def timetext_to_minutes (self, time):
            """
            Convierte "HH:MM:SS" a minutos decimales.
            """
            return self.timetext_to_seconds(time) / 60
        
        
        def timetext_to_hours (self, time):
            """
            Convierte "HH:MM:SS" a horas decimales.
            Ideal cuando el feature de entrada del modelo es "horas de estudio".
            """
            return self.timetext_to_minutes(time) / 60
    
    
    class CSV:
        """
        Lee archivos CSV y preprocesa datasets para entrenamiento.
        Se encarga de leer, barajar, dividir en train/test y separar
        features (x) de etiquetas (y).
        """
        
        @staticmethod
        def process (dataset, train_size=0.8, random_state=None, split=True):
            """
            Preprocesa un dataset: lo baraja según random_state, lo divide en
            train/test si split=True, y separa x de y. Si split=False devuelve
            solo (x, y) del dataset completo.
            """
            if random_state is not None:
                import random
                random.seed(random_state)
                random.shuffle(dataset)
            elif random_state == "shuffle" and dataset is not None and len(dataset) > 0 and isinstance(dataset[0], (list, tuple)) and len(dataset[0]) == 2:
                import random
                random.shuffle(dataset)
            elif random_state == "sort" and dataset is not None and len(dataset) > 0 and isinstance(dataset[0], (list, tuple)) and len(dataset[0]) == 2:
                dataset.sort()
            elif random_state == "reverse" and dataset is not None and len(dataset) > 0 and isinstance(dataset[0], (list, tuple)) and len(dataset[0]) == 2:
                dataset.reverse()
            elif random_state is not None:
                raise ValueError("Invalid random_state value")
            
            if split and (train_size <= 0 or train_size >= 1):
                raise ValueError("train_size must be between 0 and 1")
            
            if dataset is None or len(dataset) == 0:
                raise ValueError("dataset cannot be empty")
            
            if not split:
                return [x for x, y in dataset], [y for x, y in dataset]
            else:
                split_index = int(len(dataset) * train_size)
                train_data = dataset[:split_index]
                test_data = dataset[split_index:]
                
                x_train = [x for x, y in train_data]
                y_train = [y for x, y in train_data]
                x_test = [x for x, y in test_data]
                y_test = [y for x, y in test_data]

                return x_train, y_train, x_test, y_test
        
        
        @staticmethod
        def read (file_path, headers=False, separator=",", start_row=0):
            """
            Lee un archivo CSV y devuelve su contenido como lista de listas.
            Puede saltar encabezados y empezar desde una fila específica.
            """
            if headers:
                start_row += 1
            data = []
            with open(file_path, "r") as file:
                for i, line in enumerate(file):
                    if i >= start_row:
                        data.append(line.strip().split(separator))
            return data
    
    
    class Vector:
        """
        Implementación propia de vectores y matrices con soporte para producto
        punto, multiplicación matriz-vector, matriz-matriz y batch matmul.
        Reemplaza a numpy para mantener el proyecto sin dependencias externas
        y entender qué pasa detrás del operador @.
        """
        
        def __init__(self, data):
            self.data = data
            self.shape = self._get_shape(data)
        
        
        def _get_shape(self, data):
            """
            Calcula las dimensiones del vector/matriz recursivamente.
            Devuelve una lista como [filas, columnas] o [batch, filas, columnas].
            """
            shape = []
            curr = data
            while isinstance(curr, list):
                shape.append(len(curr))
                curr = curr[0] if len(curr) > 0 else None
            return shape
    
        
        def __matmul__(self, other):
            """
            Sobrecarga del operador @ para soportar:
            - 1D @ 1D → producto punto escalar
            - 2D @ 1D → matriz por vector (forward pass)
            - 2D @ 2D → multiplicación de matrices estándar
            - 3D @ 3D → batch matmul (múltiples matrices a la vez)
            """
            dim_a = len(self.shape)
            dim_b = len(other.shape)

            if dim_a == 1 and dim_b == 1:
                if self.shape[0] != other.shape[0]:
                    raise ValueError("Dimensiones incompatibles para dot product")
                return sum(a * b for a, b in zip(self.data, other.data))

            elif dim_a == 2 and dim_b == 1:
                if self.shape[1] != other.shape[0]:
                    raise ValueError("Columnas A deben coincidir con Filas B")
                return [sum(row[i] * other.data[i] for i in range(len(other.data))) 
                        for row in self.data]

            elif dim_a == 2 and dim_b == 2:
                return self._matmul_2d(self.data, other.data)

            elif dim_a == 3 and dim_b == 3:
                if self.shape[0] != other.shape[0]:
                    raise ValueError("Los batches deben tener el mismo tamaño")
                return [self._matmul_2d(self.data[i], other.data[i]) 
                        for i in range(self.shape[0])]
            
            else:
                raise NotImplementedError("Esa combinación de dimensiones no está soportada aún.")


        def _matmul_2d(self, A, B):
            """
            Multiplicación de matrices 2D estándar (filas de A por columnas de B).
            Es la operación base que reutilizan los demás casos.
            """
            res = [[sum(A[i][k] * B[k][j] for k in range(len(B))) 
                    for j in range(len(B[0]))] for i in range(len(A))]
            return res