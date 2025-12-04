"""
Script para cargar un modelo SVM pre-entrenado y realizar predicciones.

Uso:
    python prediccion.py modelo.h5 imagen.jpg
    python prediccion.py modelo.h5 carpeta_imagenes/
    python prediccion.py  # Usa valores por defecto

Dependencias:
    - numpy
    - h5py
    - opencv-python (cv2)
    - scikit-image
"""

import os
import sys
import numpy as np
import cv2
import h5py
import json

# Importar las clases necesarias del módulo SVM
from SVM import ClasificadorSVM


def cargar_imagen(ruta_imagen, target_size=(64, 64)):
    """
    Carga y preprocesa una imagen para predicción.
    
    Args:
        ruta_imagen (str): Ruta a la imagen.
        target_size (tuple): Tamaño de redimensionamiento.
        
    Returns:
        np.ndarray: Imagen preprocesada lista para predicción.
    """
    img = cv2.imread(ruta_imagen)
    
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
    
    # Convertir BGR a RGB (OpenCV carga en BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalizar a [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0
    
    return img_normalized


def predecir_imagen(modelo, ruta_imagen):
    """
    Realiza predicción sobre una sola imagen.
    
    Args:
        modelo: Instancia de ClasificadorSVM cargado.
        ruta_imagen (str): Ruta a la imagen.
        
    Returns:
        tuple: (clase_predicha, nombre_clase)
    """
    # Cargar y preprocesar imagen
    img = cargar_imagen(ruta_imagen, modelo.img_size)
    
    # Expandir dimensiones para batch de 1
    X = np.expand_dims(img, axis=0)
    
    # Predecir
    prediccion = modelo.predecir(X)[0]
    
    # Obtener nombre de la clase
    inv_label_map = {v: k for k, v in modelo.label_map.items()}
    nombre_clase = inv_label_map[prediccion]
    
    return prediccion, nombre_clase


def predecir_carpeta(modelo, ruta_carpeta):
    """
    Realiza predicción sobre todas las imágenes de una carpeta.
    
    Args:
        modelo: Instancia de ClasificadorSVM cargado.
        ruta_carpeta (str): Ruta a la carpeta con imágenes.
        
    Returns:
        list: Lista de tuplas (archivo, clase_id, nombre_clase)
    """
    extensiones_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    resultados = []
    
    archivos = [f for f in os.listdir(ruta_carpeta) 
                if f.lower().endswith(extensiones_validas)]
    
    print(f"\nProcesando {len(archivos)} imágenes...")
    
    for archivo in archivos:
        ruta_completa = os.path.join(ruta_carpeta, archivo)
        try:
            clase_id, nombre_clase = predecir_imagen(modelo, ruta_completa)
            resultados.append((archivo, clase_id, nombre_clase))
            print(f"   {archivo}: {nombre_clase}")
        except Exception as e:
            print(f"   Error en {archivo}: {e}")
    
    return resultados


def mostrar_info_modelo(modelo):
    """Muestra información del modelo cargado."""
    print("\n" + "="*50)
    print("INFORMACIÓN DEL MODELO CARGADO")
    print("="*50)
    print(f"   Estrategia de features: {modelo.strategy}")
    print(f"   Parámetro C: {modelo.C}")
    print(f"   Usa PCA: {modelo.use_pca}")
    if modelo.use_pca:
        print(f"   Componentes PCA: {modelo.n_components}")
    print(f"   Tamaño de imagen: {modelo.img_size}")
    print(f"   Clases disponibles: {list(modelo.label_map.keys())}")
    print("="*50 + "\n")


def main():
    """Función principal del script de predicción."""
    
    # Valores por defecto
    modelo_path = "modelo_svm_optimizado.h5"
    
    # Procesar argumentos de línea de comandos
    if len(sys.argv) >= 2:
        modelo_path = sys.argv[1]
    
    # Verificar que existe el archivo del modelo
    if not os.path.exists(modelo_path):
        print(f"❌ Error: No se encontró el modelo en: {modelo_path}")
        print("   Entrena un modelo primero ejecutando: python SVM.py")
        sys.exit(1)
    
    # Cargar modelo
    print("\n🔄 Cargando modelo...")
    modelo = ClasificadorSVM.cargar_modelo(modelo_path)
    mostrar_info_modelo(modelo)
    
    # Modo interactivo o por argumentos
    if len(sys.argv) >= 3:
        ruta_entrada = sys.argv[2]
        
        if os.path.isfile(ruta_entrada):
            # Predicción de una sola imagen
            print(f"📷 Prediciendo imagen: {ruta_entrada}")
            clase_id, nombre_clase = predecir_imagen(modelo, ruta_entrada)
            print(f"\n   🎯 Predicción: {nombre_clase} (clase {clase_id})")
            
        elif os.path.isdir(ruta_entrada):
            # Predicción de una carpeta
            print(f"📁 Prediciendo carpeta: {ruta_entrada}")
            resultados = predecir_carpeta(modelo, ruta_entrada)
            
            # Resumen
            print(f"\n📊 Total de imágenes procesadas: {len(resultados)}")
            
        else:
            print(f"❌ Error: La ruta no existe: {ruta_entrada}")
            sys.exit(1)
    else:
        # Modo demostración: probar con el conjunto de test
        print("🔬 Modo demostración: Evaluando en conjunto de test...")
        
        from Preprocesamiento import load_and_preprocess_dataset
        X_test, y_test, _ = load_and_preprocess_dataset(split='test')
        
        if X_test is not None and len(X_test) > 0:
            # Evaluar modelo
            from SVM import MetricsCalculator
            y_pred = modelo.predecir(X_test)
            
            metrics = MetricsCalculator.get_metrics(y_test, y_pred, modelo.classes)
            
            print("\n📊 RESULTADOS DE EVALUACIÓN:")
            print(f"   🎯 Accuracy:  {metrics['accuracy']*100:.2f}%")
            print(f"   🎯 Precision: {metrics['precision']*100:.2f}%")
            print(f"   🎯 Recall:    {metrics['recall']*100:.2f}%")
            print(f"   🎯 F1-Score:  {metrics['f1']*100:.2f}%")
            
            # Mostrar algunas predicciones de ejemplo
            inv_map = {v: k for k, v in modelo.label_map.items()}
            print("\n📝 Ejemplos de predicciones (primeras 10):")
            for i in range(min(10, len(y_test))):
                real = inv_map[y_test[i]]
                pred = inv_map[y_pred[i]]
                estado = "✅" if y_test[i] == y_pred[i] else "❌"
                print(f"   {estado} Real: {real:<6} | Predicción: {pred}")
        else:
            print("❌ No se pudo cargar el conjunto de test")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("   SISTEMA DE PREDICCIÓN - SVM CLASIFICADOR DE ANIMALES")
    print("="*60)
    main()
    print("\n✅ Proceso completado.\n")
