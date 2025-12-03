import os
import cv2
import numpy as np
import sys

def load_and_preprocess_dataset(split='train', target_size=(64, 64)):
    """
    Carga imágenes desde ../dataset/{split}/ respetando la estructura de carpetas.
    
    Args:
        split (str): 'train' o 'test' para elegir la subcarpeta.
        target_size (tuple): Tamaño final (ancho, alto).
    """
    
    # 1. Obtener la ruta absoluta del directorio donde está ESTE script (src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Navegar un nivel arriba (..) y entrar a dataset -> split (train/test)
    # Ruta resultante: .../100.-Examen_Redes_Neuronales/Pipeline/dataset/train
    dataset_path = os.path.join(current_dir, '..', 'dataset', split)
    
    # Verificación de seguridad
    if not os.path.exists(dataset_path):
        print(f"ERROR CRÍTICO: No se encuentra la ruta: {dataset_path}")
        return None, None, None

    # 3. Definir las clases EXACTAMENTE como se llaman tus carpetas en la imagen
    # Nota: Usamos 'deep' porque así está tu carpeta, aunque sea 'deer' (venado)
    classes = ['cat', 'cow', 'deep', 'dog', 'lion']
    
    # Mapeo para etiquetas numéricas: {'cat': 0, 'cow': 1, ...}
    label_map = {cls: i for i, cls in enumerate(classes)}
    
    images_list = []
    labels_list = []
    
    print(f"--- Cargando conjunto de datos: {split.upper()} ---")
    print(f"Ruta base: {dataset_path}")
    
    for cls_name in classes:
        cls_folder = os.path.join(dataset_path, cls_name)
        
        if not os.path.exists(cls_folder):
            print(f"⚠️ Advertencia: No se encontró la carpeta '{cls_name}' en {split}")
            continue
            
        files = os.listdir(cls_folder)
        count = 0
        
        for file_name in files:
            # Filtrar solo imágenes por extensión (buena práctica en Linux)
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            img_full_path = os.path.join(cls_folder, file_name)
            
            try:
                # Leer imagen
                img = cv2.imread(img_full_path)
                
                if img is None:
                    print(f"   Error: No se pudo leer {file_name}")
                    continue
                
                # Preprocesamiento Pipeline
                # 1. Grises
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 2. Resize
                img_resized = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)
                # 3. Normalización [0, 1]
                img_normalized = img_resized.astype('float32') / 255.0
                
                images_list.append(img_normalized)
                labels_list.append(label_map[cls_name])
                count += 1
                
            except Exception as e:
                print(f"Error en {file_name}: {e}")
                
        print(f"   Clase '{cls_name}': {count} imágenes cargadas.")

    # Convertir a NumPy arrays
    X = np.array(images_list)
    y = np.array(labels_list)
    
    # Shuffle (Mezclar datos)
    if len(X) > 0:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    return X, y, label_map
import matplotlib.pyplot as plt
if __name__ == "__main__":
    # Prueba de carga del set de entrenamiento
    print("Probando carga de TRAIN...")
    X_train, y_train, classes = load_and_preprocess_dataset(split='train')
    
    if X_train is not None and len(X_train) > 0:
        print(f"\nÉxito: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    else:
        print("\nError: No se cargaron datos de train. Revisa las carpetas.")

    print("\nProbando carga de TEST...")
    X_test, y_test, _ = load_and_preprocess_dataset(split='test')

    if X_test is not None and len(X_test) > 0:
        print(f"\nÉxito: X_train shape: {X_test.shape}, y_train shape: {y_test.shape}")
    else:
        print("\nError: No se cargaron datos de train. Revisa las carpetas.")

    plt.imshow(X_train[0], cmap='gray')
    plt.axis("off")
    plt.savefig("imagen_prueba.png")