#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:03:01 2024

@author: neves
"""

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

LABELS = ["Cachorros", "Cavalos", "Galinhas", "Gatos", "Vacas"]  # Ajuste as classes conforme necessário

# Carregar o modelo
@st.cache_resource

# Pré-processamento da imagem
def preprocess_image(uploaded_file, target_size):
    try:
        image = load_img(uploaded_file, target_size=target_size)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0) / 255.0
        return image_array
    except Exception as e:
        st.error(f"Erro ao processar a imagem: {e}")
        return None

# Prever a classe da imagem
def predict_class(image_array, model):
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return predicted_class, confidence

# Interface da aplicação
st.title("Classificador de Imagens de Animais")
st.write("Carregue uma imagem para identificar a classe correspondente.")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    model = load_model("TP3-IC/modelo_treinado.h5")
    image_array = preprocess_image(uploaded_file, target_size=(256, 256))
    if image_array is not None:
        predicted_class, confidence = predict_class(image_array, model)
        st.image(uploaded_file, caption=f"Imagem carregada", use_column_width=True)
        st.write(f"Classe prevista: **{LABELS[predicted_class]}**")
        st.write(f"Confiança: **{confidence * 100:.2f}%**")

