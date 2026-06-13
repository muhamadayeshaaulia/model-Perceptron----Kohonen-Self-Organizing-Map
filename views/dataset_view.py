import streamlit as st


def render(repo):
    """Render halaman tampilan dataset."""
    st.header("📊 Dataset Mahasiswa")
    st.write("Dataset ini digunakan sebagai data latih untuk model Perceptron dan SOM.")

    st.dataframe(repo.df, width="stretch")

    st.write("**Keterangan label:**")
    col1, col2 = st.columns(2)
    with col1:
        st.success("**1** = Produktif")
    with col2:
        st.error("**0** = Tidak Produktif")

    st.divider()
    st.write(f"**Total data:** {len(repo.df)} baris | **Fitur:** {len(repo.fitur)} kolom")
