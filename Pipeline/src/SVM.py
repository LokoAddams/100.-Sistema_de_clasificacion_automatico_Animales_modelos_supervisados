import sys
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
import h5py
import json
from scipy.optimize import minimize
# [cite_start]Librerías permitidas para visión por computadora [cite: 23, 57]
from skimage.feature import hog, local_binary_pattern
from skimage import color, transform

warnings.filterwarnings("ignore")

# =============================================================================
# 1. UTILIDADES MATEMÁTICAS Y PREPROCESAMIENTO
# =============================================================================

class CustomScaler:
    """Normalización Z-score manual (StandardScaler)."""
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.scale

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class CustomPCA:
    """PCA manual usando SVD de NumPy (Optimizado para velocidad)."""
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        # SVD Reducido (Full matrices = False)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.n_components]
        return self

    def transform(self, X):
        return np.dot(X - self.mean, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class MetricsCalculator:
    """Cálculo manual de métricas para clasificación multiclase."""
    @staticmethod
    def get_metrics(y_true, y_pred, classes):
        # [cite_start]Accuracy Global [cite: 50]
        acc = np.mean(y_true == y_pred)
        
        precisions = []
        recalls = []
        f1s = []
        
        # Calcular métricas por clase (Macro-average)
        for c in classes:
            # Definir positivos y negativos para la clase actual
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))
            
            # [cite_start]Precisión [cite: 51]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            # [cite_start]Exhaustividad (Recall) [cite: 52]
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            # [cite_start]F1 Score [cite: 53]
            f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            
        return {
            "accuracy": acc,
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1s)
        }

# =============================================================================
# 2. CORE DEL MODELO (SVM EXACTO)
# =============================================================================

class SVM_Binario_Exacto:
    """
    SVM Lineal usando scipy.optimize.minimize (L-BFGS-B).
    Minimiza la Squared Hinge Loss para diferenciabilidad y convergencia rápida.
    """
    def __init__(self, C=1.0):
        self.C = C
        self.w = None
        self.b = None

    def _loss(self, params, X, y):
        # Desempaquetar pesos y bias
        w = params[:-1]
        b = params[-1]
        
        # Margen = 1 - y * (w.x + b)
        margins = 1 - y * (np.dot(X, w) + b)
        # Squared Hinge Loss: max(0, margen)^2
        hinge = np.maximum(0, margins)
        
        # Función de costo: Regularización + Penalización C
        return 0.5 * np.dot(w, w) + self.C * np.sum(hinge ** 2)

    def _grad(self, params, X, y):
        w = params[:-1]
        b = params[-1]
        margins = 1 - y * (np.dot(X, w) + b)
        
        # Máscara para identificar vectores de soporte (los que generan error)
        mask = (margins > 0).astype(float)
        hinge = margins * mask
        
        common = -2 * self.C * hinge * y
        
        grad_w = w + np.dot(common, X)
        grad_b = np.sum(common)
        
        return np.append(grad_w, grad_b)

    def fit(self, X, y):
        dim = X.shape[1]
        initial_params = np.zeros(dim + 1)
        
        # Optimización numérica exacta
        res = minimize(
            fun=self._loss, x0=initial_params, args=(X, y),
            jac=self._grad, method='L-BFGS-B',
            options={'maxiter': 500}
        )
        self.w = res.x[:-1]
        self.b = res.x[-1]

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

# =============================================================================
# 3. CLASIFICADOR MULTICLASE Y PIPELINE UNIFICADO
# =============================================================================

class ClasificadorSVM:
    def __init__(self, C=1.0, strategy='hog', use_pca=False, n_components=150):
        self.C = C
        self.strategy = strategy
        self.use_pca = use_pca
        self.n_components = n_components
        
        self.models = {}
        self.scaler = CustomScaler()
        self.pca = None
        self.classes = []
        self.img_size = (64, 64) 
        self.label_map = {}

    def _extract_features(self, X, is_training=False):
        """
        Extracción flexible basada en la estrategia.
        Integra la lógica optimizada de SVM6 (pesos de color, HOG+LBP).
        """
        features_list = []
        if is_training:
            print(f"   [Procesando] Estrategia: {self.strategy.upper()} | PCA: {self.use_pca}")

        for img in X:
            # Normalización de entrada
            if img.max() > 1.0: img = img / 255.0
            img_res = transform.resize(img, self.img_size, anti_aliasing=True)
            
            feats = []
            
            # [cite_start]--- 1. HOG (Forma) [cite: 38] ---
            # Siempre se incluye si la estrategia tiene 'hog' o es 'full'
            if 'hog' in self.strategy or 'full' in self.strategy:
                if img_res.ndim == 3: gray = color.rgb2gray(img_res)
                else: gray = img_res
                f_hog = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
                feats.append(f_hog)

            # --- 2. COLOR (HSV) ---
            # Clave para el modelo optimizado de SVM6
            if 'color' in self.strategy or 'full' in self.strategy:
                if img_res.ndim == 3:
                    hsv = color.rgb2hsv(img_res)
                    h_h, _ = np.histogram(hsv[:,:,0], bins=16, range=(0,1), density=True)
                    h_s, _ = np.histogram(hsv[:,:,1], bins=8, range=(0,1), density=True)
                    h_v, _ = np.histogram(hsv[:,:,2], bins=8, range=(0,1), density=True)
                    f_col = np.concatenate([h_h, h_s, h_v])
                    # Peso 1.5 como en SVM6.py (Mejor Modelo)
                    feats.append(f_col * 1.5) 
                else:
                    feats.append(np.zeros(32))

            # --- 3. LBP (Textura) ---
            if 'lbp' in self.strategy or 'full' in self.strategy:
                if img_res.ndim == 3: gray = color.rgb2gray(img_res)
                else: gray = img_res
                lbp = local_binary_pattern(gray, P=16, R=2, method='uniform')
                n_bins = int(lbp.max() + 1)
                f_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, density=True)
                feats.append(f_lbp)

            features_list.append(np.concatenate(feats))

        X_feats = np.array(features_list)
        
        # Pipeline: Scaler -> PCA -> SVM
        if is_training:
            X_scaled = self.scaler.fit_transform(X_feats)
            if self.use_pca:
                n_comp = min(self.n_components, X_scaled.shape[0], X_scaled.shape[1])
                self.pca = CustomPCA(n_components=n_comp)
                X_final = self.pca.fit_transform(X_scaled)
            else:
                X_final = X_scaled
        else:
            X_scaled = self.scaler.transform(X_feats)
            if self.use_pca:
                X_final = self.pca.transform(X_scaled)
            else:
                X_final = X_scaled
                
        return X_final

    def entrenar(self, X, y, label_map):
        self.classes = np.unique(y)
        self.label_map = label_map
        
        # Preprocesamiento y extracción
        X_ready = self._extract_features(X, is_training=True)
        
        # [cite_start]Estrategia One-vs-Rest [cite: 31]
        for c in self.classes:
            y_binary = np.where(y == c, 1, -1)
            svm = SVM_Binario_Exacto(C=self.C)
            svm.fit(X_ready, y_binary)
            self.models[c] = svm
        
        self.is_trained = True

    def predecir(self, X):
        X_ready = self._extract_features(X, is_training=False)
        scores = np.zeros((X_ready.shape[0], len(self.classes)))
        for idx, c in enumerate(self.classes):
            scores[:, idx] = self.models[c].decision_function(X_ready)
        return self.classes[np.argmax(scores, axis=1)]

    def evaluar(self, X_test, y_test, nombre_exp=""):
        y_pred = self.predecir(X_test)
        
        # [cite_start]Calcular todas las métricas requeridas [cite: 48-54]
        metrics = MetricsCalculator.get_metrics(y_test, y_pred, self.classes)
        
        # Generar Matriz de Confusión Gráfica
        self._plot_confusion_matrix(y_test, y_pred, nombre_exp)
        
        return metrics

    def _plot_confusion_matrix(self, y_true, y_pred, title):
        """Dibuja y guarda la matriz de confusión."""
        n_classes = len(self.classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            idx_t = np.where(self.classes == t)[0][0]
            idx_p = np.where(self.classes == p)[0][0]
            cm[idx_t, idx_p] += 1
            
        inv_map = {v: k for k, v in self.label_map.items()}
        names = [inv_map[c] for c in self.classes]

        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{title}')
        plt.colorbar()
        tick_marks = np.arange(len(names))
        plt.xticks(tick_marks, names, rotation=45)
        plt.yticks(tick_marks, names)

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        filename = f"cm_{title.replace(' ', '_').replace('.','')}.png"
        plt.savefig(filename)
        plt.close()
        print(f"   📊 Matriz guardada: {filename}")

    def guardar_modelo(self, filepath):
        """
        Guarda todos los parámetros del modelo en un archivo HDF5.
        
        Args:
            filepath (str): Ruta del archivo .h5 donde guardar el modelo.
        """
        if not hasattr(self, 'is_trained') or not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Entrena primero con .entrenar()")
        
        with h5py.File(filepath, 'w') as f:
            # --- 1. Metadata del modelo ---
            meta = f.create_group('metadata')
            meta.attrs['C'] = self.C
            meta.attrs['strategy'] = self.strategy
            meta.attrs['use_pca'] = self.use_pca
            meta.attrs['n_components'] = self.n_components
            meta.attrs['img_size'] = self.img_size
            
            # Guardar clases
            f.create_dataset('classes', data=self.classes)
            
            # Guardar label_map como JSON string
            meta.attrs['label_map'] = json.dumps(self.label_map)
            
            # --- 2. Scaler (Normalizador Z-score) ---
            scaler_grp = f.create_group('scaler')
            scaler_grp.create_dataset('mean', data=self.scaler.mean)
            scaler_grp.create_dataset('scale', data=self.scaler.scale)
            
            # --- 3. PCA (si se usó) ---
            if self.use_pca and self.pca is not None:
                pca_grp = f.create_group('pca')
                pca_grp.create_dataset('components', data=self.pca.components)
                pca_grp.create_dataset('mean', data=self.pca.mean)
            
            # --- 4. Modelos SVM (pesos w y bias b para cada clase) ---
            models_grp = f.create_group('svm_models')
            for clase, svm in self.models.items():
                clase_grp = models_grp.create_group(f'clase_{clase}')
                clase_grp.create_dataset('w', data=svm.w)
                clase_grp.create_dataset('b', data=np.array([svm.b]))
        
        print(f"   💾 Modelo guardado exitosamente en: {filepath}")

    @classmethod
    def cargar_modelo(cls, filepath):
        """
        Carga un modelo previamente guardado desde un archivo HDF5.
        
        Args:
            filepath (str): Ruta del archivo .h5 con el modelo guardado.
            
        Returns:
            ClasificadorSVM: Instancia del clasificador con los pesos cargados.
        """
        with h5py.File(filepath, 'r') as f:
            # --- 1. Cargar metadata ---
            meta = f['metadata']
            C = meta.attrs['C']
            strategy = meta.attrs['strategy']
            use_pca = meta.attrs['use_pca']
            n_components = meta.attrs['n_components']
            img_size = tuple(meta.attrs['img_size'])
            label_map = json.loads(meta.attrs['label_map'])
            
            # Crear instancia con los parámetros originales
            modelo = cls(C=C, strategy=strategy, use_pca=use_pca, n_components=n_components)
            modelo.img_size = img_size
            modelo.label_map = label_map
            modelo.classes = f['classes'][:]
            
            # --- 2. Cargar Scaler ---
            modelo.scaler = CustomScaler()
            modelo.scaler.mean = f['scaler/mean'][:]
            modelo.scaler.scale = f['scaler/scale'][:]
            
            # --- 3. Cargar PCA (si existe) ---
            if 'pca' in f:
                modelo.pca = CustomPCA(n_components=n_components)
                modelo.pca.components = f['pca/components'][:]
                modelo.pca.mean = f['pca/mean'][:]
            else:
                modelo.pca = None
            
            # --- 4. Cargar modelos SVM ---
            models_grp = f['svm_models']
            for key in models_grp.keys():
                clase = int(key.split('_')[1])
                svm = SVM_Binario_Exacto(C=C)
                svm.w = models_grp[key]['w'][:]
                svm.b = models_grp[key]['b'][0]
                modelo.models[clase] = svm
            
            modelo.is_trained = True
        
        print(f"   ✅ Modelo cargado exitosamente desde: {filepath}")
        return modelo



# =============================================================================
# 4. EJECUCIÓN EXPERIMENTAL COMPLETA
# =============================================================================
if __name__ == "__main__":
    from Preprocesamiento import load_and_preprocess_dataset
    
    print("\n=== CARGA DE DATOS ===")
    X_train, y_train, label_map = load_and_preprocess_dataset(split='train')
    X_test, y_test, _ = load_and_preprocess_dataset(split='test')

    X_train_flipped = np.array([np.fliplr(img) for img in X_train])
    
    # Concatenar originales + espejo
    X_train = np.concatenate([X_train, X_train_flipped], axis=0)
    y_train = np.concatenate([y_train, y_train], axis=0)
    


    # Lista de 4 experimentos (Los 3 originales + El Optimizador SVM6)
    experimentos = [
        {
            'nombre': '1. Baseline (Solo HOG)',
            'params': {'C': 1.0, 'strategy': 'hog', 'use_pca': False}
        },
        {
            'nombre': '2. Intermedio (HOG + LBP)',
            'params': {'C': 1.0, 'strategy': 'hog+lbp', 'use_pca': False}
        },
        {
            'nombre': '3. Completo sin PCA',
            'params': {'C': 1.0, 'strategy': 'hog+lbp+color', 'use_pca': False}
        },
        {
            # REPLICA EXACTA DE TU "MEJOR MODELO" (SVM6.py)
            'nombre': '4. MODELO OPTIMIZADO (PCA + L-BFGS-B)',
            'params': {
                'C': 5.0,                 # Penalización más alta
                'strategy': 'hog+lbp+color', # Todas las feats (con peso 1.5 en color)
                'use_pca': True,          # PCA activado
                'n_components': 150       # Reducción drástica para velocidad
            }
        }
    ]

    print("\n=== INICIANDO ANÁLISIS COMPARATIVO (4 EXPERIMENTOS) ===")
    resultados_finales = []

    for exp in experimentos:
        print(f"\n>> {exp['nombre']}")
        modelo = ClasificadorSVM(**exp['params'])
        
        # Entrenamiento con medición de tiempo
        start = time.time()
        modelo.entrenar(X_train, y_train, label_map)
        train_time = time.time() - start
        
        # Evaluación con todas las métricas
        metrics = modelo.evaluar(X_test, y_test, exp['nombre'])
        
        print(f"   ✅ Tiempo Entrenamiento: {train_time:.2f}s")
        print(f"   🎯 Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"   🎯 Precision: {metrics['precision']*100:.2f}%")
        print(f"   🎯 Recall:    {metrics['recall']*100:.2f}%")
        print(f"   🎯 F1-Score:  {metrics['f1']*100:.2f}%")
        
        resultados_finales.append((exp['nombre'], metrics['accuracy'], metrics['f1']))

    print("\n" + "="*60)
    print(f"{'EXPERIMENTO':<35} | {'ACCURACY':<10} | {'F1-SCORE':<10}")
    print("="*60)
    for nombre, acc, f1 in resultados_finales:
        print(f"{nombre:<35} | {acc*100:.2f}%     | {f1*100:.2f}%")

    # Guardar el mejor modelo (Experimento 4 - Modelo Optimizado)
    print("\n=== GUARDANDO MEJOR MODELO ===")
    mejor_modelo = ClasificadorSVM(**experimentos[3]['params'])
    mejor_modelo.entrenar(X_train, y_train, label_map)
    mejor_modelo.guardar_modelo("modelo_svm_optimizado.h5")