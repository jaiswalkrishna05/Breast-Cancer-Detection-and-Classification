"""
streamlit_app.py - Web application for Cancer Detection and Classification

This Streamlit app provides a simple user interface for uploading cancer
histology images and obtaining a prediction from a trained CNN model.
The user can choose between the baseline CNN and the transfer learning
(ResNet50) model.  After uploading an image, the app displays the
original image, preprocesses it in the same way as during training,
predicts the probability of the image being malignant, and reports the
predicted class along with the probability.

To run the app:
    streamlit run streamlit_app.py

The app assumes that the trained model files ``baseline_cnn.keras`` and
``transfer_resnet50.keras`` are located in a ``models/`` folder
relative to where the script is executed.
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Constants
IMG_SIZE = (224, 224)
CLASS_NAMES = ['benign', 'malignant']
MODEL_OPTIONS = {
    'Baseline CNN': 'models/baseline_cnn.keras',
    'Transfer Learning (ResNet50)': 'models/transfer_resnet50.keras'
}

# Page configuration
st.set_page_config(page_title='Cancer Detection (CNN)', layout='centered')
st.title('Cancer Detection and Classification')
st.write('Upload a histology image to predict whether it is benign or malignant.')

# Model selection
model_choice = st.selectbox('Select a model', list(MODEL_OPTIONS.keys()))
model_path = MODEL_OPTIONS[model_choice]

@st.cache_resource
def load_model(path):
    """Loads a Keras model from disk.  Cached to avoid reloading on every
    interaction."""
    return tf.keras.models.load_model(path)

model = load_model(model_path)

# Image uploader
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image_resized = image.resize(IMG_SIZE)
    img_array = np.array(image_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    probability = model.predict(img_array, verbose=0)[0][0]
    predicted_class = 1 if probability >= 0.5 else 0

    # Display results
    st.subheader('Prediction Results')
    st.write(f'**Predicted Class:** {CLASS_NAMES[predicted_class]}')
    st.write(f'**Malignant Probability:** {probability:.4f}')
else:
    st.write('Please upload an image to get a prediction.')

