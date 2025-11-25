# Neural_Network_From_Skratch
# MNIST Digit Classifier

Proyecto Final - Inteligencia Artificial en Videojuegos  
Universidad Panamericana

## Archivos del Proyecto

- `neuron.py` - Clase Neurona
- `layer.py` - Clase Capa
- `ann.py` - Red Neuronal Artificial
- `main.py` - Script principal del clasificador MNIST
- `requirements.txt` - Dependencias

## Instalación

```bash
pip install -r requirements.txt
```

Esto instala: numpy, matplotlib, tensorflow

## Ejecución

```bash
python main.py
```

El programa:
1. Carga el dataset MNIST
2. Entrena una red neuronal con backpropagation
3. Evalúa el modelo en datos de prueba
4. Genera visualizaciones en la carpeta `results/`

## Configuración

Para modificar parámetros, edita las variables en `main.py`:

```python
TRAIN_SIZE = 5000          # Muestras de entrenamiento
TEST_SIZE = 1000           # Muestras de prueba
HIDDEN_LAYERS = 2          # Número de capas ocultas
NEURONS_PER_LAYER = 128    # Neuronas por capa
LEARNING_RATE = 0.1        # Tasa de aprendizaje
EPOCHS = 20                # Épocas de entrenamiento
```

## Resultados

El programa genera en la carpeta `results/`:
- `training_history.png` - Gráficas de pérdida y precisión
- `predictions.png` - Visualización de predicciones

## Arquitectura de la Red

- **Entrada:** 784 neuronas (imágenes 28×28 pixels)
- **Capa Oculta 1:** 128 neuronas
- **Capa Oculta 2:** 128 neuronas
- **Salida:** 10 neuronas (dígitos 0-9)
- **Activación:** Sigmoid
- **Algoritmo:** Backpropagation

## Precisión Esperada

Con la configuración por defecto:
- Entrenamiento: ~95-98%
- Validación: ~93-96%
- Prueba: ~93-96%

Tiempo de entrenamiento: ~10-15 minutos