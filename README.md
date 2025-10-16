# GAN para Reconstrucción de Señales EMG

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.7+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1007/978--3--030--68790--8__19-brightgreen)](https://doi.org/10.1007/978-3-030-68790-8_19)

Una implementación robusta de **CycleGAN** para la transformación de señales EMG entre dominios de sujetos sanos y amputados, utilizando los datasets Ninapro DB1 y DB3.

## 🚀 Características

- **Arquitectura CycleGAN** para traducción unidireccional de señales biomédicas
- **Preprocesamiento robusto** de señales EMG con normalización y ventaneo
- **Generadores con skip connections** (U-Net like) para preservar información
- **Entrenamiento estabilizado** con pérdidas de ciclo e identidad
- **Compatibilidad** con datasets Ninapro DB1 (sanos) y DB3 (amputados)

## 📋 Requisitos

```bash
tensorflow>=2.4.0
numpy>=1.19.0
scipy>=1.6.0
matplotlib>=3.3.0
```

## 🏗️ Estructura del Modelo

### Generador (U-Net Like)
```python
Input (200, 12) → Encoder (Conv1D) → Bottleneck → Decoder (Conv1DTranspose) → Output (200, 12)
```

### Discriminador
```python
Input (200, 12) → Conv1D (64) → Conv1D (128) → Conv1D (256) → Output (1)
```

## 📊 Datasets

| Dataset | Sujetos | Condición | Canales EMG |
|---------|---------|-----------|-------------|
| **Ninapro DB1** | 27 | Sanos | 12 |
| **Ninapro DB3** | 11 | Amputados | 12 |

## 🛠️ Configuración

```python
WINDOW_SIZE = 200      # Tamaño de ventana temporal
NUM_CHANNELS = 12      # Canales EMG
BATCH_SIZE = 32        # Tamaño del lote
EPOCHS = 50            # Épocas de entrenamiento
LEARNING_RATE = 2e-4   # Tasa de aprendizaje
```

## 🚀 Uso

### Entrenamiento
```python
python gan_training_script.py
```

### Generación de Señales
```python
# Cargar modelo entrenado
generator = tf.keras.models.load_model('generador_gan.h5')

# Transformar señal de amputado a "sano"
reconstructed_signal = generator.predict(amputado_signal)
```

## 📈 Funciones de Pérdida

- **Pérdida del Generador**: Binary Crossentropy
- **Pérdida del Discriminador**: Binary Crossentropy
- **Pérdida de Ciclo**: MAE (×10)
- **Pérdida de Identidad**: MAE (×5)

## 💾 Salida

El modelo guarda:
- `generador_gan.h5` - Modelo final entrenado
- `generador_gan_epoch_{N}.h5` - Checkpoints cada 5 épocas
- Gráficos de comparación señales originales/reconstruidas

## 🎯 Aplicación

Este modelo está diseñado para:
- **Investigación en prótesis inteligentes**
- **Traducción de señales EMG** entre dominios
- **Data augmentation** para clasificación de gestos
- **Estudios de rehabilitación** y interfaces cerebro-máquina

## 📄 Referencias

```bibtex
@inproceedings{gan_emg_2021,
  title={CycleGAN for EMG Signal Domain Translation},
  author={Research Team},
  booktitle={International Conference on Biomedical Engineering},
  year={2021}
}
```

## 👥 Contribución

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Contacto

Para preguntas sobre este proyecto, por favor abra un issue en el repositorio.

---

**Nota**: Este código está diseñado para investigación académica. Para uso clínico, se requieren validaciones adicionales.
