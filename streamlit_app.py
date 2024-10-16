
import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the model
@st.cache_resource
#funtion to load the model
def load_trained_model():
    return load_model('model.h5')

model = load_trained_model()

# Class names (placeholder, replace with actual class names)
class_names = ["biotite", "bornite", "chrysocolla" , "malachite" , "muscovite" ,"pyrite" ,"quartz"]  # Modify this as per your model's classes

# Function to make predictions
def predict_image(img):
    img = cv2.resize(img, (32, 32))  # Resize to match the input size of the model
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return prediction

# Streamlit app layout
st.title('Minerals Classification Application')

# Upload image
uploaded_file = st.file_uploader("Upload the mineral's image and the system will classify it, classes are: biotite, bornite, chrysocolla, malachite, muscovite, pyrite, quartz", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = image.load_img(uploaded_file, target_size=(32, 32))
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess and predict
    img_array = image.img_to_array(img)
    prediction = predict_image(img_array)

    # Get the predicted class
    index = np.argmax(prediction) 
    predicted_class = class_names[index]
    
    # Display the prediction
    st.write(f'Predicted Class: {predicted_class}')

    
