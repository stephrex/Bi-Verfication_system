import streamlit as st

st.title("Test File Upload")

file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if file:
    st.image(file)
