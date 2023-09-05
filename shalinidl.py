import numpy as np
import pandas as pd
import tensorflow 
import keras
#import scikit-learn
from keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
#from scikit-learn.preprocessing import MinMaxScaler
import streamlit as st
import cv2

# Load the trained model
model =tensorflow. keras.Sequential()
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.load_weights('model.h5')

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (28, 28))
    return resized_image

def predict_digit(image):
    processed_image = preprocess_image(image)
    processed_image = processed_image.reshape(1, 28, 28, 1)
    pred = model.predict(processed_image)
    pred_lst = pred[0, :].tolist()
    max_prob = max(pred_lst)
    pred_idx = pred_lst.index(max_prob)
    return pred_idx

def main():
    st.title("Digit Recognition with Streamlit")

    # Capture an image from the webcam
    st.header("Capture an Image")
    capture_button = st.button("Capture Image")
    if capture_button:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open the camera.")
        else:
            ret, frame = cap.read()
            if ret:
                cv2.imwrite("captured_image.jpg", frame)
                st.image("captured_image.jpg", caption="Captured Image", use_column_width=True)
                st.success("Image captured and saved as captured_image.jpg")
            cap.release()

    # Process and recognize the captured image
    st.header("Recognize Digit")
    recognize_button = st.button("Recognize Digit")
    if recognize_button:
        image = cv2.imread("captured_image.jpg")
        if image is not None:
            predicted_digit = predict_digit(image)
            st.success(f"Model predicts the digit is {predicted_digit}")
        else:
            st.error("Error: Image not found or couldn't be loaded.")

if __name__ == "__main__":
    main()
