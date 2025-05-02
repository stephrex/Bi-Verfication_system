import streamlit as st

st.title("Test File Upload")

file = st.file_uploader("Upload image")
if file:
    st.image(file)
