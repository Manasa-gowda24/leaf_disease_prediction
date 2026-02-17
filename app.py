import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# -------------------------------
# Google Drive Model Download
# -------------------------------

file_id = "1stsbfe2yvG5sG2eUY9h0oSeSoi1J923i"
model_path = "trained_plant_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(id=file_id, output=model_path, quiet=False)

# -------------------------------
# Load Model
# -------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

try:
    model = load_model()
except Exception as e:
    st.error("Model loading failed. Check your model file.")
    st.stop()

# -------------------------------
# Prediction Function
# -------------------------------

def model_prediction(test_image):
    image = Image.open(test_image).resize((128,128))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# -------------------------------
# Load Top Image
# -------------------------------

base_path = os.path.dirname(__file__)
image_path = os.path.join(base_path, "Disease.png")

if os.path.exists(image_path):
    img = Image.open(image_path)
    st.image(img)
else:
    st.warning("Disease.png not found in project folder")

# -------------------------------
# HOME PAGE
# -------------------------------

if app_mode == "HOME":
    st.markdown(
        "<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>",
        unsafe_allow_html=True
    )

# -------------------------------
# DISEASE RECOGNITION PAGE
# -------------------------------

elif app_mode == "DISEASE RECOGNITION":

    st.header("Upload Leaf Image for Disease Detection")

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:

        st.image(test_image, use_column_width=True)

        if st.button("Predict"):

            with st.spinner("Predicting..."):
                result_index = model_prediction(test_image)

                class_name = ['Early_Blight', 'Healthy', 'Late_Blight']

                st.success(f"Model Prediction: {class_name[result_index]}")
