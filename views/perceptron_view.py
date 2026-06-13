import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.styles import apply_custom_styles


def render(repo, perceptron):
    """Render halaman hasil training Perceptron."""
    apply_custom_styles()
    st.header("📉 Hasil Training Perceptron")

    st.subheader("Kurva Penurunan Error per Epoch")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    # Modern premium dark console style
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#1E293B')
    epochs = range(1, len(perceptron.errors) + 1)
    ax.plot(
        epochs,
        perceptron.errors,
        marker="o",
        color="#818CF8",     # Glowing Indigo
        linewidth=2.5,
        markersize=6,
        markerfacecolor="#C084FC", # Violet glow
        markeredgecolor="#0F172A",
        markeredgewidth=1.5,
        label="Total Error"
    )
    ax.set_xlabel("Epoch", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
    ax.set_ylabel("Total Error", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
    ax.set_title("Kurva Penurunan Error Perceptron", fontsize=12, fontweight='bold', color='#F8FAFC', pad=15)
    # Grid and spines configuration
    ax.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    st.pyplot(fig)

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("⚖️ Bobot Akhir Fitur")
        bobot_df = pd.DataFrame({"Fitur": repo.fitur, "Bobot": perceptron.weights})
        st.dataframe(bobot_df, width="stretch")
    with col2:
        st.subheader("🎯 Parameter Model")
        st.metric("Bias Akhir", round(float(perceptron.bias), 4))
    with col3:
        st.subheader("📈 Akurasi Model")
        predictions = perceptron.predict(repo.X)
        accuracy = (predictions == repo.y).mean() * 100
        st.metric("Akurasi Training", f"{accuracy:.2f}%")

    st.divider()
    st.subheader("🔍 Langkah Perhitungan Klasifikasi Perceptron Per Baris Data")
    st.write("Pilih salah satu baris indeks data latih untuk melihat simulasi perhitungan kombinasi linear (Net Input) dan fungsi aktivasi step model Perceptron.")

    # Dropdown select box
    row_idx = st.selectbox(
        "Pilih Baris Data Latih:",
        options=range(len(repo.df)),
        format_func=lambda i: f"Baris {i+1} (Tidur: {repo.df.iloc[i]['jam_tidur']}, Mood: {repo.df.iloc[i]['mood']}, Stres: {repo.df.iloc[i]['stres']}, Belajar: {repo.df.iloc[i]['jam_belajar']}, HP: {repo.df.iloc[i]['jam_hp']}, Tugas: {repo.df.iloc[i]['jumlah_tugas']} | Label: {'Produktif' if repo.df.iloc[i]['label'] == 1 else 'Tidak Produktif'})"
    )

    data_raw = repo.X[row_idx]
    target_val = repo.y[row_idx]
    target_name = "Produktif" if target_val == 1 else "Tidak Produktif"

    # Perhitungan linier Perceptron
    bobot = perceptron.weights
    bias = perceptron.bias
    kontribusi = data_raw * bobot
    net_input = np.sum(kontribusi) + bias
    prediksi_val = 1 if net_input >= 0 else 0
    prediksi_name = "Produktif" if prediksi_val == 1 else "Tidak Produktif"
    error = target_val - prediksi_val

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.write("**Data Aktivitas Terpilih:**")
        st.dataframe(pd.DataFrame([data_raw], columns=repo.fitur), width="stretch")
    with col_p2:
        st.write("**Hasil Evaluasi:**")
        eval_df = pd.DataFrame({
            "Metrik": ["Target Kelas Asli", "Prediksi Model", "Error (Target - Prediksi)", "Status"],
            "Nilai": [f"{target_val} ({target_name})", f"{prediksi_val} ({prediksi_name})", str(error), "Sesuai ✅" if error == 0 else "Tidak Sesuai ❌"]
        })
        st.dataframe(eval_df, width="stretch")

    # Tampilkan langkah perhitungan matematis
    with st.expander("🔍 Rincian Perhitungan Matematika (Kombinasi Linear & Aktivasi)"):
        st.markdown("**1. Rumus Net Input (Kombinasi Linear):**")
        st.latex(r"y_{in} = \sum_{i=1}^6 x_i w_i + b")
        st.markdown("**2. Substitusi Nilai Fitur & Bobot Akhir:**")
        langkah_perkalian = " + ".join([f"({data_raw[i]:.1f} \\times {bobot[i]:.4f})" for i in range(len(bobot))])
        st.latex(rf"y_{{in}} = {langkah_perkalian} + ({bias:.4f})")
        st.latex(rf"y_{{in}} = {np.sum(kontribusi):.4f} + ({bias:.4f}) = {net_input:.4f}")
        st.markdown("**3. Aplikasi Fungsi Aktivasi (Step Function):**")
        st.latex(r"y = f(y_{in}) = \begin{cases} 1 & \text{jika } y_{in} \ge 0 \\ 0 & \text{jika } y_{in} < 0 \end{cases}")
        status_aktivasi = "\\ge 0" if net_input >= 0 else "< 0"
        st.latex(rf"y = f({net_input:.4f}) \rightarrow y_{{prediksi}} = {prediksi_val} \text{{ ({prediksi_name})}}")

    if error == 0:
        st.success(f"**Kesimpulan:** Net Input = **{net_input:.4f}** ({status_aktivasi.replace('\\', '')}), mengaktifkan fungsi step menjadi **{prediksi_name}**, sesuai dengan target kelas asli **{target_name}**.")
    else:
        st.error(f"**Kesimpulan:** Net Input = **{net_input:.4f}** ({status_aktivasi.replace('\\', '')}), mengaktifkan fungsi step menjadi **{prediksi_name}**, berbeda dengan target kelas asli **{target_name}**.")

    st.divider()
    st.subheader("📖 Teori & Rumus Perhitungan Perceptron")

    st.markdown(f"""
    Model JST Perceptron adalah algoritma *Supervised Learning* yang digunakan untuk klasifikasi biner. Selama proses training, bobot ($w$) dan bias ($b$) diperbarui terus-menerus menggunakan rumus berikut:

    ##### 1. Persamaan Net Input (Kombinasi Linear)
    $$y_{{in}} = \\sum_{{i=1}}^n x_i w_i + b$$

    ##### 2. Fungsi Aktivasi (Undak Biner / Step Function)
    $$y = f(y_{{in}}) = \\begin{{cases}} 1 & \\text{{jika }} y_{{in}} \\ge 0 \\\\ 0 & \\text{{jika }} y_{{in}} < 0 \\end{{cases}}$$

    ##### 3. Rumus Pembaruan Bobot dan Bias (Delta Rule)
    Jika terdapat error ($e = y_{{\\text{{target}}}} - y_{{\\text{{prediksi}}}}$) antara target dan prediksi:
    $$\\Delta w_i = \\alpha \\times e \\times x_i$$
    $$\\Delta b = \\alpha \\times e$$

    Setelah delta dihitung, bobot dan bias baru diperbarui dengan:
    $$w_i \\leftarrow w_i + \\Delta w_i$$
    $$b \\leftarrow b + \\Delta b$$

    *Keterangan:*
    *   $x_i$: Nilai fitur ke-$i$ (Jam Tidur, Mood, Stres, Belajar, HP, Tugas)
    *   $w_i$: Bobot fitur ke-$i$ (Bobot akhir ditampilkan pada tabel di atas)
    *   $b$: Bias model (Bias akhir ditampilkan pada kolom di atas)
    *   $\\alpha$: *Learning Rate* (pada model ini diatur sebesar `{perceptron.learning_rate}`)
    *   $e$: Error (selisih target dengan prediksi)
    """)

