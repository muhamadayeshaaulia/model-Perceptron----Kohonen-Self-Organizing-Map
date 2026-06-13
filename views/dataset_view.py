import streamlit as st
from utils.styles import apply_custom_styles


def render(repo):
    """Render halaman tampilan dataset."""
    apply_custom_styles()
    st.header("📊 Dataset Mahasiswa")
    st.write("Dataset ini digunakan sebagai data latih untuk melatih model Perceptron dan Kohonen (SOM).")

    st.dataframe(repo.df, width="stretch")

    st.markdown("### 🏷️ Keterangan Label target")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div style="background-color: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.15); border-left: 5px solid #10B981; padding: 1.25rem; border-radius: 12px; height: 100%; color: var(--text-color);">
                <h4 style="margin: 0 0 8px 0; color: #10B981; font-size: 1.1rem; font-family: 'Outfit';">🟢 Label 1: Produktif</h4>
                <p style="margin: 0; opacity: 0.9; font-size: 0.9rem; line-height: 1.5; color: var(--text-color);">
                    Menunjukkan kondisi aktivitas harian mahasiswa yang seimbang, efisien, dengan istirahat cukup, tingkat stres rendah-sedang, dan belajar produktif.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div style="background-color: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.15); border-left: 5px solid #EF4444; padding: 1.25rem; border-radius: 12px; height: 100%; color: var(--text-color);">
                <h4 style="margin: 0 0 8px 0; color: #EF4444; font-size: 1.1rem; font-family: 'Outfit';">🔴 Label 0: Tidak Produktif</h4>
                <p style="margin: 0; opacity: 0.9; font-size: 0.9rem; line-height: 1.5; color: var(--text-color);">
                    Menunjukkan kondisi rentan burnout atau kurang produktif, sering kali dipicu oleh jam tidur kurang, stres tinggi, tugas menumpuk, atau jam main HP yang berlebih.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown(
        f"""
        <div style="text-align: center; color: #64748B; font-size: 0.9rem; font-weight: 500;">
            📊 Total Data: <b>{len(repo.df)}</b> baris | ⚙️ Jumlah Fitur: <b>{len(repo.fitur)}</b> kolom
        </div>
        """,
        unsafe_allow_html=True
    )

