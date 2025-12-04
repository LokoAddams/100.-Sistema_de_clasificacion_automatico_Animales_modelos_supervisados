=== CARGA DE DATOS ===
--- Cargando conjunto de datos: TRAIN ---
Ruta base: /home/lucas-alcoba/Documents/1.-Universidad/6.-SextoSemestre/Modelado/5.-MachineLearning/100.-Sistema_de_clasificacion_automatico_Animales_modelos_supervisados/Pipeline/src/../dataset/train
   Clase 'cat': 91 imágenes cargadas.
   Clase 'cow': 86 imágenes cargadas.
   Clase 'deep': 87 imágenes cargadas.
   Clase 'dog': 111 imágenes cargadas.
   Clase 'lion': 89 imágenes cargadas.
--- Cargando conjunto de datos: TEST ---
Ruta base: /home/lucas-alcoba/Documents/1.-Universidad/6.-SextoSemestre/Modelado/5.-MachineLearning/100.-Sistema_de_clasificacion_automatico_Animales_modelos_supervisados/Pipeline/src/../dataset/test
   Clase 'cat': 17 imágenes cargadas.
   Clase 'cow': 16 imágenes cargadas.
   Clase 'deep': 16 imágenes cargadas.
   Clase 'dog': 16 imágenes cargadas.
   Clase 'lion': 17 imágenes cargadas.

=== INICIANDO ANÁLISIS COMPARATIVO (4 EXPERIMENTOS) ===

>> 1. Baseline (Solo HOG)
   [Procesando] Estrategia: HOG | PCA: False
   📊 Matriz guardada: cm_1_Baseline_(Solo_HOG).png
   ✅ Tiempo Entrenamiento: 81.69s
   🎯 Accuracy:  32.93%
   🎯 Precision: 33.41%
   🎯 Recall:    33.01%
   🎯 F1-Score:  33.11%

>> 2. Intermedio (HOG + LBP)
   [Procesando] Estrategia: HOG+LBP | PCA: False
   📊 Matriz guardada: cm_2_Intermedio_(HOG_+_LBP).png
   ✅ Tiempo Entrenamiento: 66.99s
   🎯 Accuracy:  37.80%
   🎯 Precision: 38.32%
   🎯 Recall:    38.01%
   🎯 F1-Score:  37.85%

>> 3. Completo sin PCA
   [Procesando] Estrategia: HOG+LBP+COLOR | PCA: False
   📊 Matriz guardada: cm_3_Completo_sin_PCA.png
   ✅ Tiempo Entrenamiento: 69.51s
   🎯 Accuracy:  37.80%
   🎯 Precision: 38.32%
   🎯 Recall:    38.01%
   🎯 F1-Score:  37.85%

>> 4. MODELO OPTIMIZADO (PCA + L-BFGS-B)
   [Procesando] Estrategia: HOG+LBP+COLOR | PCA: True
   📊 Matriz guardada: cm_4_MODELO_OPTIMIZADO_(PCA_+_L-BFGS-B).png
   ✅ Tiempo Entrenamiento: 2.23s
   🎯 Accuracy:  53.66%
   🎯 Precision: 55.77%
   🎯 Recall:    53.75%
   🎯 F1-Score:  53.80%

============================================================
EXPERIMENTO                         | ACCURACY   | F1-SCORE  
============================================================
1. Baseline (Solo HOG)              | 32.93%     | 33.11%
2. Intermedio (HOG + LBP)           | 37.80%     | 37.85%
3. Completo sin PCA                 | 37.80%     | 37.85%
4. MODELO OPTIMIZADO (PCA + L-BFGS-B) | 53.66%     | 53.80%