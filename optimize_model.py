import tensorflow as tf
from keras import models, datasets
import numpy as np

#insira seu código aqui

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [input_value]

# Carregamento do modelo treinado

model = models.load_model("model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Conversão para TensorFlow Lite (.tflite)
# Aplicação de técnica de otimização, como: Dynamic Range Quantization

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)