import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model.keras")

st.set_page_config(page_title="MNIST Digit Recognition", layout="centered")

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a 28x28 grayscale image of a digit (0-9)")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

def preprocess_image(img):
    img = img.convert("L")              # grayscale
    img = img.resize((28, 28))         # resize to MNIST size
    img = np.array(img)

    # Normalize
    img = img / 255.0

    # Reshape for CNN
    img = img.reshape(1, 28, 28, 1)
    return img

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    processed_img = preprocess_image(image)

    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)

    st.subheader(f"🧠 Predicted Digit: {predicted_class}")
    st.write("Confidence:", np.max(prediction))