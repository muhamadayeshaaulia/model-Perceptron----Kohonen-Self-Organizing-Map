import streamlit as st
from services.model_service import load_model
from views.som_view import render

st.set_page_config(page_title="Clustering SOM", page_icon="🧩", layout="wide")
repo, perceptron, som = load_model()
render(repo, som)
