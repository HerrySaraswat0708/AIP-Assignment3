import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from model_loader import load_model
from utils import preprocess, postprocess

st.title("Zero-DCE Low Light Enhancement")

model = load_model()

uploaded_file = st.file_uploader("Upload Low Light Image")

if uploaded_file:
    
    image = Image.open(uploaded_file)
    image = np.array(image)
    H,W,C = image.shape
    st.subheader("Original Image")
    st.image(image)

    if st.button("Enhance Image"):
        
        img_tensor = preprocess(image)

        with torch.no_grad():
            _,enhanced, _ = model(img_tensor)

        result = postprocess(enhanced)
        result = cv2.resize(result,(H,W))
        st.subheader("Enhanced Image")
        st.image(result)