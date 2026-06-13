import streamlit as st
from utils.styles import apply_custom_styles


def render():
    """Render halaman tentang aplikasi."""
    apply_custom_styles()
    st.header("ℹ️ Tentang Aplikasi")

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(124, 58, 237, 0.08) 100%); border: 1px solid rgba(79, 70, 229, 0.2); padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem;">
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6; color: var(--text-color); font-family: 'Plus Jakarta Sans'; opacity: 0.95;">
                Aplikasi ini dikembangkan sebagai proyek <b>Ujian Akhir Semester (UAS)</b> untuk mata kuliah <b>Jaringan Syaraf Tiruan</b>. 
                Tujuannya adalah membantu mahasiswa menganalisis kondisi produktivitas harian dan mendeteksi potensi kejenuhan (burnout) 
                berdasarkan pola aktivitas harian menggunakan dua model JST yang saling melengkapi.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="custom-card">
                <h3 style="color: #4F46E5; font-size: 1.3rem; display: flex; align-items: center; gap: 8px;">
                    🧠 Model JST yang Digunakan
                </h3>
                <div style="margin-bottom: 1.25rem;">
                    <strong style="color: #4F46E5; font-size: 1rem;">1. Perceptron (Supervised Classification)</strong>
                    <p style="margin: 4px 0 0 0;">
                        Berfungsi untuk klasifikasi biner — mengidentifikasi apakah kondisi harian termasuk kategori <b>Produktif</b> atau <b>Tidak Produktif</b>.
                    </p>
                </div>
                <div>
                    <strong style="color: #7C3AED; font-size: 1rem;">2. Kohonen Self-Organizing Map (SOM) (Unsupervised Clustering)</strong>
                    <p style="margin: 4px 0 0 0;">
                        Berfungsi untuk clustering data tanpa label — mengelompokkan pola aktivitas ke dalam 3 klaster kondisi mahasiswa:
                    </p>
                    <ul style="margin: 8px 0 0 0; padding-left: 20px;">
                        <li>🟢 <span style="color: #10B981; font-weight: 600;">Produktif</span>: Aktivitas optimal & seimbang.</li>
                        <li>🔵 <span style="color: #3B82F6; font-weight: 600;">Santai</span>: Beban rendah, istirahat berlebih.</li>
                        <li>🔴 <span style="color: #EF4444; font-weight: 600;">Burnout</span>: Beban tugas dan stres tinggi, kurang tidur.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="custom-card" style="justify-content: space-between !important;">
                <div>
                    <h3 style="color: #7C3AED; font-size: 1.3rem; display: flex; align-items: center; gap: 8px;">
                        📥 Input Sistem (Aktivitas Harian)
                    </h3>
                    <ul style="margin: 0 0 1.5rem 0; padding-left: 20px;">
                        <li>🛌 <b>Jam tidur</b> (durasi istirahat malam)</li>
                        <li>😊 <b>Mood pagi</b> (tingkat suasana hati awal hari)</li>
                        <li>😤 <b>Tingkat stres</b> (beban tekanan emosional)</li>
                        <li>📚 <b>Jam belajar</b> (waktu studi produktif)</li>
                        <li>📱 <b>Jam main HP</b> (waktu screen-time non-produktif)</li>
                        <li>📝 <b>Jumlah tugas</b> (beban akademis aktif)</li>
                    </ul>
                </div>
                <div>
                    <h3 style="color: #EC4899; font-size: 1.3rem; display: flex; align-items: center; gap: 8px;">
                        📤 Output Analisis
                    </h3>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>📊 Status produktivitas (Perceptron)</li>
                        <li>🧩 Kategori klaster burnout (Kohonen SOM)</li>
                        <li>💡 Saran rekomendasi personal untuk perbaikan aktivitas</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

