import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os


st.set_page_config(page_title="Face & Fingerprint Verification",
                   page_icon="🔍", layout="centered")

_PROJECT_DIR = os.getcwd()


class L1DistanceLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return tf.abs(x1 - x2)

# Load model


def load_model():
    print("Loading model...")
    model = tf.keras.models.load_model(
        # Adjust path if needed
        _PROJECT_DIR + '/siamese_model2.keras',
        custom_objects={"L1DistanceLayer": L1DistanceLayer},
        safe_mode=False
    )
    print("Model loaded successfully!")
    return model


siamese_model = load_model()

# Preprocessing function


def load_and_preprocess_image(uploaded_file, target_size=(128, 128)):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array


# Main app
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Biometric Verification System 🔒</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a Face and a Fingerprint to Verify Identity</p>",
            unsafe_allow_html=True)

# Upload images
col1, col2 = st.columns(2)
with col1:
    face_image = st.file_uploader(
        "Upload Face Image 🧑",
        type=["jpg", "jpeg", "png", "bmp"]
    )
with col2:
    fingerprint_image = st.file_uploader(
        "Upload Fingerprint Image 🖐️",
        type=["jpg", "jpeg", "png", "bmp"]
    )

# Predict button
if st.button("Verify Identity 🚀"):
    if face_image is not None and fingerprint_image is not None:
        # Preprocess images
        face_img = load_and_preprocess_image(face_image)
        fp_img = load_and_preprocess_image(fingerprint_image)

        face_img = np.expand_dims(face_img, axis=0)
        fp_img = np.expand_dims(fp_img, axis=0)

        # Predict
        prediction = siamese_model.predict([face_img, fp_img])[0][0]

        st.markdown("---")
        st.subheader("Verification Result:")

        if prediction > 0.5:
            st.success(f"✅ MATCH FOUND! (Similarity Score: {prediction:.4f})")
            st.markdown(
                "<h2 style='text-align: center; color: green;'>🟢 Identity Verified</h2>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.error(f"❌ NO MATCH (Similarity Score: {prediction:.4f})")
            st.markdown(
                "<h2 style='text-align: center; color: red;'>🔴 Identity Mismatch</h2>", unsafe_allow_html=True)
            st.snow()

    else:
        st.warning("Please upload both a face and a fingerprint image!")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 12px;'>Developed by Your Iremide ~ The AI Guy</p>",
            unsafe_allow_html=True)
