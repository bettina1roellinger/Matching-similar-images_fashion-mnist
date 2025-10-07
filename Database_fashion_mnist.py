import tensorflow as tf

(x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.fashion_mnist.load_data()
X_valid = x_train_full[:5000]
y_valid = y_train_full[:5000]
X_train = x_train_full[5000:]
y_train = y_train_full[5000:]

import numpy as np
X_train = np.expand_dims(X_train, axis=-1)
X_valid = np.expand_dims(X_valid, axis=-1)
X_test = np.expand_dims(x_test_full, axis=-1) 

X_train = X_train / 255.0
X_test = X_test / 255.0
X_valid= X_valid / 255.0
print("Min:", X_train.min())
print("Max:", X_train.max())
print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)


from build_embedding_model import load_embedding_model
embedding_model = load_embedding_model("fashion_mnist.keras")
embeddings = embedding_model.predict(X_train, batch_size=64)  # shape (60000, 64)

import sqlite3
import os
from PIL import Image

conn = sqlite3.connect("images.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT,
    label INTEGER,
    embedding BLOB
)
""")

os.makedirs("data", exist_ok=True)

for i, (img, label, emb) in enumerate(zip(X_train, y_train, embeddings)):
    filename = f"data/img_{i:05d}.png"

    # Sauvegarder l’image
    im = Image.fromarray((img.squeeze() * 255).astype(np.uint8))
    im.save(filename)

    # Insérer dans la base
    cursor.execute(
        "INSERT INTO images (path, label, embedding) VALUES (?, ?, ?)",
        (filename, int(label), emb.astype(np.float32).tobytes())
    )

conn.commit()
conn.close()