import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.styles import apply_custom_styles


def render(repo, perceptron):
    """Render halaman hasil training Perceptron."""
    apply_custom_styles()
    st.header("📉 Hasil Training Perceptron")

    # --- Tampilkan Dataset & Hasil Prediksi ---
    st.subheader("📋 Data Latih & Hasil Prediksi Model")
    df_pred = repo.df.copy()
    predictions = perceptron.predict(repo.X)
    df_pred["Prediksi_Model"] = ["Produktif" if p == 1 else "Tidak Produktif" for p in predictions]
    df_pred["Target_Asli"] = ["Produktif" if l == 1 else "Tidak Produktif" for l in df_pred["label"]]
    df_pred["Status"] = ["✅ Benar" if p == l else "❌ Salah" for p, l in zip(predictions, df_pred["label"])]
    # Reorder columns to put Target_Asli, Prediksi_Model, Status at the end
    cols = [c for c in df_pred.columns if c not in ["label", "Target_Asli", "Prediksi_Model", "Status"]] + ["Target_Asli", "Prediksi_Model", "Status"]
    st.dataframe(df_pred[cols], width="stretch")
    st.divider()
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

    # Tabel epoch vs error
    st.write("##### 📋 Tabel Error per Epoch")
    df_epoch = pd.DataFrame({
        "Epoch": list(range(1, len(perceptron.errors) + 1)),
        "Total Error": perceptron.errors,
        "Status": ["\u2705 Konvergen (Error = 0)" if e == 0 else "⏳ Masih ada error" for e in perceptron.errors]
    })
    st.dataframe(df_epoch, width="stretch", hide_index=True)

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

    with st.expander("Bagaimana cara kerja bobot dan bias Perceptron?"):
        st.write("Bobot dan bias di atas bukanlah angka yang ditebak, melainkan hasil dari **proses training (pembelajaran) iteratif** Perceptron menggunakan data latih Anda.")
        st.write("Pada setiap epoch, jika model melakukan kesalahan prediksi pada suatu baris data, ia akan mengoreksi nilai bobot dan biasnya menggunakan rumus **Perceptron Learning Rule**:")
        st.latex(r"\Delta w = \alpha \times (y_{\text{target}} - y_{\text{prediksi}}) \times x")
        st.latex(r"w_{\text{baru}} = w_{\text{lama}} + \Delta w")
        st.latex(r"b_{\text{baru}} = b_{\text{lama}} + \alpha \times (y_{\text{target}} - y_{\text{prediksi}})")

        st.markdown(
            """
            **Keterangan:**
            - $\\alpha$ (Alpha): **Learning Rate** (Laju Pembelajaran), menentukan seberapa besar langkah koreksi.
            - $y_{\\text{target}}$: Label asli dari dataset (1 untuk Produktif, 0 untuk Tidak Produktif).
            - $y_{\\text{prediksi}}$: Prediksi model saat itu (hasil dari fungsi aktivasi).
            - $x$: Nilai fitur input (misal: jam tidur, tingkat stres, dll).
            **Contoh Logika:**
            Jika model menebak "Produktif" (1) padahal aslinya "Tidak Produktif" (0), maka $(0 - 1) = -1$.
            Bobot fitur yang bernilai positif akan **dikurangi** agar tebakan selanjutnya menjadi lebih akurat. Proses ini terus diulang (lihat *Tabel Error per Epoch* di atas) hingga total error mencapai 0 atau target iterasi (epoch) maksimum tercapai.
            """
        )

    with st.expander("🔍 Lihat Detail Step-by-Step Perubahan Bobot Selama Training"):
        st.write("Berikut ini adalah log atau rekaman **setiap kali bobot diperbarui** (karena prediksi model meleset) selama proses training:")
        
        # Simulasi ulang untuk mencatat history
        sim_w = np.zeros(repo.X.shape[1])
        sim_b = 0.0
        alpha = perceptron.learning_rate
        history = []
        
        for ep in range(perceptron.epoch):
            error_count = 0
            for i, (xi, target) in enumerate(zip(repo.X, repo.y)):
                net_input = np.dot(xi, sim_w) + sim_b
                pred = 1 if net_input >= 0 else 0
                err = target - pred
                if err != 0:
                    old_w = sim_w.copy()
                    sim_w += alpha * err * xi
                    sim_b += alpha * err
                    error_count += 1
                    
                    # Rekam ke history
                    hist_row = {
                        "Epoch": ep + 1,
                        "Baris Data": i + 1,
                        "Target": target,
                        "Prediksi": pred,
                        "Error": err,
                        "Bias Baru": round(sim_b, 4)
                    }
                    for j, fit_name in enumerate(repo.fitur):
                        hist_row[f"Bobot_{fit_name}"] = round(sim_w[j], 4)
                    history.append(hist_row)
            if error_count == 0:
                break # Sudah konvergen
                
        df_hist = pd.DataFrame(history)
        if not df_hist.empty:
            st.dataframe(df_hist, width="stretch", hide_index=True)
            st.caption(f"*Hanya baris data yang mengalami error (meleset) yang memicu perubahan bobot dan bias. Total update: {len(df_hist)} kali.*")
        else:
            st.info("Bobot awal sudah langsung memberikan prediksi yang benar (tidak ada perubahan).")

    st.divider()
    st.subheader("📊 Visualisasi Garis Keputusan (Decision Boundary) Perceptron")
    st.write("Garis pemisah Perceptron pada dimensi **Mood vs Stres**, menggunakan nilai rata-rata fitur lainnya dari data latih.")

    fig_db, ax_db = plt.subplots(figsize=(9, 5))
    fig_db.patch.set_facecolor('#0F172A')
    ax_db.set_facecolor('#1E293B')

    colors_db = {0: "#F87171", 1: "#34D399"}
    for label_id in [0, 1]:
        data_label = repo.df[repo.df["label"] == label_id]
        ax_db.scatter(
            data_label["mood"], data_label["stres"],
            label="Produktif (Data Latih)" if label_id == 1 else "Tidak Produktif (Data Latih)",
            color=colors_db[label_id], alpha=0.85,
            edgecolors="#1E293B", linewidths=0.7, s=85, zorder=3
        )

    # Hitung garis keputusan menggunakan nilai rata-rata fitur non-plot
    w = perceptron.weights
    b = perceptron.bias
    # index: 0=tidur, 1=mood, 2=stres, 3=belajar, 4=hp, 5=tugas
    mean_tidur   = float(repo.df["jam_tidur"].mean())
    mean_belajar = float(repo.df["jam_belajar"].mean())
    mean_hp      = float(repo.df["jam_hp"].mean())
    mean_tugas   = float(repo.df["jumlah_tugas"].mean())
    const_val = b + w[0]*mean_tidur + w[3]*mean_belajar + w[4]*mean_hp + w[5]*mean_tugas

    mood_range = np.linspace(repo.nilai_min[1], repo.nilai_max[1], 200)
    if w[2] != 0:
        stres_db = -(w[1] * mood_range + const_val) / w[2]
        ax_db.plot(mood_range, stres_db, color="#818CF8", linestyle="--", linewidth=2.5,
                   label="Garis Keputusan Perceptron", zorder=4)
    elif w[1] != 0:
        ax_db.axvline(x=-const_val/w[1], color="#818CF8", linestyle="--", linewidth=2.5,
                      label="Garis Keputusan Perceptron", zorder=4)

    ax_db.set_xlabel("Mood Pagi", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
    ax_db.set_ylabel("Tingkat Stres", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
    ax_db.set_title("Decision Boundary Perceptron (Mood vs Stres)", fontsize=12, fontweight='bold', color='#F8FAFC', pad=14)
    ax_db.legend(frameon=True, facecolor='#1E293B', edgecolor='#334155', fontsize=9)
    ax_db.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")
    ax_db.spines['top'].set_visible(False)
    ax_db.spines['right'].set_visible(False)
    ax_db.spines['left'].set_color('#334155')
    ax_db.spines['bottom'].set_color('#334155')
    ax_db.tick_params(colors='#94A3B8', labelsize=9)
    ax_db.set_ylim(0, 11)
    st.pyplot(fig_db)
    st.caption(f"*Nilai rata-rata fitur pendukung yang digunakan: Jam Tidur={mean_tidur:.1f}, Jam Belajar={mean_belajar:.1f}, Jam HP={mean_hp:.1f}, Jumlah Tugas={mean_tugas:.1f}*")

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

