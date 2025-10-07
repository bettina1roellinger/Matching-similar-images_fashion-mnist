# app.py
import streamlit as st
import sqlite3
import numpy as np
from PIL import Image
import tensorflow as tf
from numpy.linalg import norm
from build_embedding_model import load_embedding_model
import os

# === Charger le modèle d'embedding (tronqué jusqu'à Dense(64))
embedding_model = load_embedding_model("/workspace/notebooks/fashion_mnist.keras")
base_dir = "/workspace/notebooks" 
# === Fonction pour obtenir l'embedding d'une image requête
def get_embedding(img_array):
    img_array = np.expand_dims(img_array, axis=0)  # (1, 28, 28, 1)
    return embedding_model.predict(img_array, verbose=0)[0]  # (64,)

# === Fonction pour trouver les k images les plus proches
def find_similar(query_embedding, k=5):
    cursor.execute("SELECT path, label, embedding FROM images")
    rows = cursor.fetchall()

    paths, labels, embeddings = [], [], []
    for path, label, emb_blob in rows:
        #paths.append(path)
        labels.append(label)
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        embeddings.append(emb)
        # Concaténation du chemin complet
        full_path = os.path.join(base_dir, path)
        paths.append(full_path)

    embeddings = np.array(embeddings)
    distances = norm(embeddings - query_embedding, axis=1)
    top_k_indices = np.argsort(distances)[:k]

    return [(paths[i], labels[i], distances[i]) for i in top_k_indices]

# === Connexion à la base de données
conn = sqlite3.connect("/workspace/notebooks/images.db")
cursor = conn.cursor()

# === Interface Streamlit
st.title("🔍 Recherche d'images similaires (Fashion MNIST)")

uploaded_file = st.file_uploader("🖼️ Téléverse une image (28x28 pixels, niveau de gris)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Charger et prétraiter l'image
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    img_array = np.array(img).reshape(28, 28, 1) / 255.0

    st.image(img_array.squeeze(), caption="Image requête", use_column_width=False)

    # Obtenir l'embedding
    query_embedding = get_embedding(img_array)

    # Rechercher les plus proches
    k = st.slider("Nombre d'images similaires à afficher", min_value=1, max_value=10, value=5)
    results = find_similar(query_embedding, k=k)

    st.subheader("🧠 Images les plus similaires :")
    
    cols = st.columns(len(results))  # crée autant de colonnes que d’images
    for col, (path, label, distance) in zip(cols, results):
        with col:
            st.image(path, caption=f"Label : {label} | Distance : {distance:.4f}", width=100)

    
    #for path, label, distance in results:
        #st.image(path, caption=f"Label : {label} | Distance : {distance:.4f}", width=100)

### streamlit run /workspace/notebooks/app_fashion_mnist.py --server.address=0.0.0.0 --server.port=8501 (lancer depuis terminal)


#### FONCTION POUR AJOUTER UNE NOUVELLE IMAGE A LA BASE DE DONEE ET LUI AJOUTER UN LABEL ####
#### Voir app_fashion_mnist_2.py ####