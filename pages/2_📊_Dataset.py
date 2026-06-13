import streamlit as st
from model import load_model

st.set_page_config(page_title="Dataset", page_icon="📊", layout="wide")

st.header("📊 Dataset Mahasiswa")
st.write("Dataset ini digunakan sebagai data latih untuk model Perceptron dan SOM.")

df, fitur, X, y, X_norm, nilai_min, nilai_max, perceptron, som = load_model()

st.dataframe(df, use_container_width=True)

st.write("**Keterangan label:**")
col1, col2 = st.columns(2)
with col1:
    st.success("**1** = Produktif")
with col2:
    st.error("**0** = Tidak Produktif")

st.divider()
st.write(f"**Total data:** {len(df)} baris | **Fitur:** {len(fitur)} kolom")
