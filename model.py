import streamlit as st
import pandas as pd
from perceptron import Perceptron
from som import KohonenSOM
from helper import normalisasi


@st.cache_resource
def load_model():
    """Load dataset, train Perceptron dan SOM. Di-cache agar tidak di-train ulang setiap page load."""
    df = pd.read_csv("dataset.csv")

    fitur = ["jam_tidur", "mood", "stres", "jam_belajar", "jam_hp", "jumlah_tugas"]
    X = df[fitur].values
    y = df["label"].values

    X_norm, nilai_min, nilai_max = normalisasi(X)

    perceptron = Perceptron(learning_rate=0.1, epoch=50)
    perceptron.fit(X, y)

    som = KohonenSOM(jumlah_cluster=3, learning_rate=0.5, epoch=100)
    som.fit(X_norm)

    return df, fitur, X, y, X_norm, nilai_min, nilai_max, perceptron, som
