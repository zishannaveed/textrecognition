#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import model_from_json
import gdown

# Load the model architecture from the saved JSON file
with open('model_architecture1.json', 'r') as f:
#with open('model_architecture.json', 'r') as f:
    model_json = f.read()
model = model_from_json(model_json)

# Load the model weights from the saved hdf5 file
#model.load_weights('C:/Users/Fujitsu/sgdo-15000r-30e-11943t-1331v.hdf5')

## Download the model weights file from the Google Drive link
url = 'https://drive.google.com/uc?id=1XQzYH8s0Kxzl5CItNF9r9g4jpv4ogCn2'
output = 'model_weights.hdf5'
gdown.download(url, output, quiet=False)

# Load the model weights from the downloaded file
model.load_weights('model_weights.hdf5')

char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" 

# Define the function to preprocess the image
def process_image(img):
    """
    Converts image to shape (64, 800, 1) & normalize
    """
    w, h = img.shape

    # Aspect Ratio Calculation
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)

    img = np.expand_dims(img, axis=2)

    # Normalize 
    img = img / 255

    return img

# Define the Streamlit app
def main():
    st.title("Image to Text Converter")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 0)

        # Preprocess the image
        img = process_image(img)

        # Reshape the image to fit the model input shape
        img = np.reshape(img, (1, 32, 128, 1))

        # Make the prediction using the loaded model
        prediction = model.predict(img)
        decoded = K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1], greedy=True)[0][0]
        out = K.get_value(decoded)

        # Display the predicted text
        st.subheader("Prediction:")
        predicted_text = ""
        for p in out[0]:
            if int(p) != -1:
                predicted_text += char_list[int(p)]
        st.write(predicted_text)

        # Display the uploaded image
        st.subheader("Uploaded Image:")
        st.image(uploaded_file, use_column_width=True)

if __name__ == "__main__":
    main()

