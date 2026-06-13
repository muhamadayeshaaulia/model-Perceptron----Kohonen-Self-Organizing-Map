import streamlit as st
from services.model_service import load_model
from views.perceptron_view import render

st.set_page_config(page_title="Training Perceptron", page_icon="📉", layout="wide")
repo, perceptron, som = load_model()
render(repo, perceptron)
