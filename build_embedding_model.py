# build_embedding_model.py
import tensorflow as tf

def load_embedding_model(model_path="fashion_mnist.keras"):
    model = tf.keras.models.load_model(model_path)

    # Recréer le modèle tronqué (jusqu'à Dense(64))
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = inputs
    for layer in model.layers[:-2]:  # enlever Dropout + Dense(10)
        x = layer(x)

    embedding_model = tf.keras.Model(inputs=inputs, outputs=x)
    _ = embedding_model(tf.zeros((1, 28, 28, 1)))  # build
    return embedding_model