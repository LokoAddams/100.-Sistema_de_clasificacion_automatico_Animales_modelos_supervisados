import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import joblib

class ClasificadorSVM:
    def __init__(self, kernel_type='rbf', C=1.0, gamma='scale'):
        """
        Constructor del modelo.
        
        Args:
            kernel_type (str): 'linear', 'poly', 'rbf', 'sigmoid'.
                               Define cómo trazar la frontera.
            C (float): Parámetro de regularización. 
                       Mayor C = menos tolerancia a errores en entrenamiento (riesgo de overfitting).
            gamma (str/float): Coeficiente para kernels 'rbf', 'poly'. 
                               'scale' es recomendado por defecto.
        """
        # Inicializamos el modelo de scikit-learn
        self.model = svm.SVC(kernel=kernel_type, C=C, gamma=gamma)
        self.kernel_type = kernel_type
        self.classes_map = None # Guardaremos el mapa de etiquetas aquí
        self.is_trained = False

    def _preparar_datos(self, X):
        """
        Método interno para asegurar que X sea 2D.
        Si recibe imágenes (N, 64, 64), las aplana a (N, 4096).
        Si recibe HOG/Texturas (N, features), las deja igual.
        """
        # Si X tiene más de 2 dimensiones (ej: Num_imgs, Alto, Ancho), aplanar.
        if X.ndim > 2:
            # reshape(cantidad_ejemplos, -1) colapsa el resto de dimensiones
            X_flat = X.reshape(X.shape[0], -1)
            return X_flat
        return X

    def entrenar(self, X, y, label_map):
        """
        Entrena el modelo con los datos proporcionados.
        
        Args:
            X (numpy array): Datos de entrada. Puede ser (N, 64, 64) o (N, features).
            y (numpy array): Etiquetas numéricas.
            label_map (dict): El diccionario que mapea {'cat': 0, ...} para reportes.
        """
        print(f"\n--- Iniciando entrenamiento SVM (Kernel: {self.kernel_type.upper()}) ---")
        
        # 1. Adaptar dimensiones si es necesario (Aplanar imágenes)
        X_ready = self._preparar_datos(X)
        self.classes_map = label_map
        
        # Guardamos los nombres de clases en orden para el reporte
        # Invertimos el diccionario: {0: 'cat', 1: 'cow'...}
        self.idx_to_class = {v: k for k, v in label_map.items()}
        
        print(f"   Input shape original: {X.shape}")
        print(f"   Input shape para SVM: {X_ready.shape}")
        
        # 2. Entrenar (Fit)
        self.model.fit(X_ready, y)
        self.is_trained = True
        print("✅ Modelo entrenado exitosamente.")

    def evaluar(self, X_test, y_test):
        """
        Realiza predicciones sobre un set de prueba y muestra métricas.
        """
        if not self.is_trained:
            print("Error: El modelo no ha sido entrenado todavía.")
            return

        X_test_ready = self._preparar_datos(X_test)
        
        print(f"\n--- Evaluando modelo en {len(X_test)} muestras ---")
        y_pred = self.model.predict(X_test_ready)
        
        # Obtener nombres de las clases ordenados por su índice numérico
        target_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        
        acc = accuracy_score(y_test, y_pred)
        print(f"Precisión Global: {acc*100:.2f}%")
        print("\nReporte Detallado:")
        print(classification_report(y_test, y_pred, target_names=target_names))

    def predecir_uno(self, x_sample):
        """
        Recibe UN solo ejemplo (matriz imagen o vector), lo adapta y predice.
        Retorna el nombre de la clase (ej: 'lion').
        """
        # Convertimos a batch de 1 elemento si viene solo
        if isinstance(x_sample, list):
            x_sample = np.array(x_sample)
            
        # Si falta la dimensión del batch (ej: shape (64,64)), agregamos una dimension
        if x_sample.ndim == 2 and self.model.n_features_in_ > x_sample.shape[1]: 
             # Caso imagen suelta (64,64) -> (1, 64, 64)
             x_sample = np.expand_dims(x_sample, axis=0)
        elif x_sample.ndim == 1:
             # Caso vector suelto (4096,) -> (1, 4096)
             x_sample = x_sample.reshape(1, -1)

        X_ready = self._preparar_datos(x_sample)
        pred_idx = self.model.predict(X_ready)[0]
        return self.idx_to_class[pred_idx]

# Importas tu función del archivo que subiste
from Preprocesamiento import load_and_preprocess_dataset
# Importas la clase que te acabo de dar
# from Modelos import ClasificadorSVM (si lo guardaste en otro lado)

if __name__ == "__main__":
    # 1. CARGA DE DATOS (Usando tu código existente)
    print("Cargando datos...")
    X_train, y_train, label_map = load_and_preprocess_dataset(split='train')
    X_test, y_test, _ = load_and_preprocess_dataset(split='test')

    if X_train is None:
        print("No se pudieron cargar los datos.")
        exit()

    # 2. INSTANCIAR EL MODELO
    # Aquí decides el kernel pasando un string: 'linear', 'rbf', 'poly'
    mi_svm = ClasificadorSVM(kernel_type='rbf', C=10)

    # 3. ENTRENAR
    # Nota: Tu X_train es (N, 64, 64). La clase lo aplanará automáticamente.
    # Si tus compañeros te pasan HOG (N, features), la clase también lo aceptará sin cambios.
    mi_svm.entrenar(X_train, y_train, label_map)

    # 4. EVALUAR
    mi_svm.evaluar(X_test, y_test)