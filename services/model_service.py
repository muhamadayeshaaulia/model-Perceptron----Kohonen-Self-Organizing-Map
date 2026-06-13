import streamlit as st
from repositories.dataset_repository import DatasetRepository
from models.perceptron import Perceptron
from models.som import KohonenSOM


@st.cache_resource
def load_model():
    """Load dataset dan training model. Di-cache agar tidak diulang tiap page load."""
    repo = DatasetRepository().load()

    perceptron = Perceptron(learning_rate=0.1, epoch=50)
    perceptron.fit(repo.X, repo.y)

    som = KohonenSOM(jumlah_cluster=3, learning_rate=0.5, epoch=100)
    som.fit(repo.X_norm)

    return repo, perceptron, som
