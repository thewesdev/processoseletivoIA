import tensorflow as tf
from keras import datasets, layers, Sequential

#insira seu código aqui

tf.config.set_visible_devices([], "GPU")

with tf.device("/CPU:0"):
    # Carregamento do dataset MNIST via TensorFlow

    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # Construção e treinamento de um modelo de classificação baseado em Redes Neurais Convolucionais (CNN)

    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = Sequential([
        layers.InputLayer(shape=(28, 28)),
        layers.Reshape(target_shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(16, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Treinamento do modelo

    model.fit(train_images, train_labels, epochs=3, validation_data=(test_images, test_labels))
    

    # Exibição da acurácia final no terminal

    _, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print("\nTest Accuracy: ", test_acc)

    # Salvamento do modelo treinado no formato Keras (.h5)

    model.save("model.h5")
