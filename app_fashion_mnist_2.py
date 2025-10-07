import streamlit as st
import numpy as np
from PIL import Image
import sqlite3
import os

# Chemin vers le dossier de travail
base_dir = "/workspace/notebooks"
db_path = os.path.join(base_dir, "images.db")
image_dir = os.path.join(base_dir, "data")

# Assure que le dossier data existe
os.makedirs(image_dir, exist_ok=True)

# Charger le modèle d'embedding
# embedding_model = keras.models.load_model("...") ← à toi d’adapter

st.header("Ajouter une nouvelle image à la base")

uploaded_file = st.file_uploader("Choisir une image PNG", type=["png", "jpg"])
label = st.number_input("Label de l'image", min_value=0, max_value=9, step=1)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    img_array = np.array(image) / 255.0  # Normalisation
    img_array = img_array.reshape(1, 28, 28, 1)

    st.image(image, caption="Image chargée", width=100)

    if st.button("Ajouter à la base"):
        # Générer l'embedding
        embedding = embedding_model.predict(img_array, batch_size=1)[0]

        # Générer un nom unique pour l’image
        existing_files = os.listdir(image_dir)
        next_id = len(existing_files)
        filename = f"img_{next_id:05d}.png"
        full_path = os.path.join(image_dir, filename)

        # Sauvegarder l’image
        image.save(full_path)

        # Sauvegarder dans la base
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO images (path, label, embedding) VALUES (?, ?, ?)
        """, (
            f"data/{filename}",  # chemin relatif
            int(label),
            embedding.astype(np.float32).tobytes()
        ))
        conn.commit()
        conn.close()

        st.success(f"Image enregistrée sous {filename} avec label {label}")