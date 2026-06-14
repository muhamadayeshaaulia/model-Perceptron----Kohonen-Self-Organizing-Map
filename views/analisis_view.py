import streamlit as st
import numpy as np
import pandas as pd
from utils.helper import normalisasi_data_baru, get_saran, cek_konsistensi
from utils.styles import apply_custom_styles
import matplotlib.pyplot as plt
from st_keyup import st_keyup  # type: ignore


def sanitize_numeric_input(val_str):
    """Menyaring karakter input agar hanya menyisakan angka dan maksimal satu pemisah desimal (. atau ,)."""
    if not val_str:
        return ""
    cleaned = []
    has_decimal = False
    for char in val_str:
        if char.isdigit():
            cleaned.append(char)
        elif char in ".," and not has_decimal:
            cleaned.append(char)
            has_decimal = True
    return "".join(cleaned)


def keyup_number_input(label, key):
    """Fungsi helper untuk mendapatkan input angka secara real-time. Memblokir huruf secara fisik."""
    val_to_use = ""
    if key in st.session_state:
        raw_val = st.session_state[key]
        if raw_val is not None:
            raw_val_str = str(raw_val)
            sanitized = sanitize_numeric_input(raw_val_str)
            if sanitized != raw_val_str:
                st.session_state[key] = sanitized
            val_to_use = sanitized

    val_str = st_keyup(label, value=val_to_use, key=key)
    if not val_str or val_str.strip() == "":
        return None, False
    
    # Jika hanya berisi titik atau koma (sedang mengetik desimal)
    if val_str.strip() in [".", ","]:
        return None, False
        
    try:
        cleaned_str = val_str.strip().replace(",", ".")
        return float(cleaned_str), False
    except ValueError:
        return None, True


def render(repo, perceptron, som):
    """Render halaman input & analisis produktivitas mahasiswa."""
    apply_custom_styles()
    st.header("📌 Input Data Harian Mahasiswa")
    st.write("Masukkan data aktivitas harian kamu secara mengetik. Validasi akan langsung merespons secara real-time pada setiap karakter yang diketik.")

    # Kolom input angka (st_keyup) kosong secara default dan merespons per keypress
    col1, col2 = st.columns(2)
    with col1:
        jam_tidur, tidur_invalid = keyup_number_input("🛌 Jam Tidur (3–8 jam)", key="jam_tidur")
        if jam_tidur is not None:
            if jam_tidur < repo.nilai_min[0] or jam_tidur > repo.nilai_max[0]:
                st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[0])} - {int(repo.nilai_max[0])})</p>", unsafe_allow_html=True)
        mood, mood_invalid = keyup_number_input("😊 Mood Pagi (3–9)", key="mood")
        if mood is not None:
            if mood < repo.nilai_min[1] or mood > repo.nilai_max[1]:
                st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[1])} - {int(repo.nilai_max[1])})</p>", unsafe_allow_html=True)
        stres, stres_invalid = keyup_number_input("😤 Tingkat Stres (1–10)", key="stres")
        if stres is not None:
            if stres < repo.nilai_min[2] or stres > repo.nilai_max[2]:
                st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[2])} - {int(repo.nilai_max[2])})</p>", unsafe_allow_html=True)

    with col2:
        jam_belajar, belajar_invalid = keyup_number_input("📚 Jam Belajar (1–6 jam)", key="jam_belajar")
        if jam_belajar is not None:
            if jam_belajar < repo.nilai_min[3] or jam_belajar > repo.nilai_max[3]:
                st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[3])} - {int(repo.nilai_max[3])})</p>", unsafe_allow_html=True)
        jam_hp, hp_invalid = keyup_number_input("📱 Jam Main HP (1–9 jam)", key="jam_hp")
        if jam_hp is not None:
            if jam_hp < repo.nilai_min[4] or jam_hp > repo.nilai_max[4]:
                st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[4])} - {int(repo.nilai_max[4])})</p>", unsafe_allow_html=True)
        jumlah_tugas, tugas_invalid = keyup_number_input("📝 Jumlah Tugas (1–8)", key="jumlah_tugas")
        if jumlah_tugas is not None:
            if jumlah_tugas < repo.nilai_min[5] or jumlah_tugas > repo.nilai_max[5]:
                st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[5])} - {int(repo.nilai_max[5])})</p>", unsafe_allow_html=True)

    # Cek apakah ada inputan yang kosong
    ada_yang_kosong = (
        jam_tidur is None or
        mood is None or
        stres is None or
        jam_belajar is None or
        jam_hp is None or
        jumlah_tugas is None
    )

    # Cek apakah ada format input yang invalid (misal diisi huruf)
    ada_yang_invalid = (
        tidur_invalid or
        mood_invalid or
        stres_invalid or
        belajar_invalid or
        hp_invalid or
        tugas_invalid
    )

    # Inisialisasi variabel untuk pengecekan fisik
    is_physically_impossible = False
    total_jam = 0.0
    if not ada_yang_kosong:
        total_jam = (jam_tidur if jam_tidur is not None else 0.0) + (jam_belajar if jam_belajar is not None else 0.0) + (jam_hp if jam_hp is not None else 0.0)
        is_physically_impossible = total_jam > 24

    # Status keaktifan tombol: nonaktif jika kosong, ada format invalid, atau jika total jam tidak valid (> 24)
    is_disabled = ada_yang_kosong or ada_yang_invalid or is_physically_impossible

    # Tampilkan info jika masih ada yang kosong
    if ada_yang_invalid:
        st.error("❌ **Format Input Salah!** Pastikan semua kolom hanya diisi dengan angka saja.")
    elif ada_yang_kosong:
        st.info("💡 **Silakan isi semua data harian di atas dengan angka untuk memulai analisis.**")
    elif is_physically_impossible:
        st.error(f"❌ **Data Input Tidak Valid secara Fisik!** Jumlah Jam Tidur ({jam_tidur} jam), Jam Belajar ({jam_belajar} jam), dan Jam HP ({jam_hp} jam) berjumlah **{total_jam} jam**, melebihi 24 jam dalam sehari. Silakan sesuaikan kembali input Anda.")

    # Pengecekan peringatan konsistensi logis (Non-blocking Warnings) jika tidak kosong, tidak invalid, dan fisik valid
    if not ada_yang_kosong and not ada_yang_invalid and not is_physically_impossible:
        pesan_inconsistent = cek_konsistensi(jam_tidur, mood, stres, jam_belajar, jam_hp, jumlah_tugas)
        pesan_inconsistent_filtered = [p for p in pesan_inconsistent if "24 jam" not in p]

        if pesan_inconsistent_filtered:
            st.warning("❌ **Data Input Tidak Konsisten!** Silakan sesuaikan kembali input Anda agar logis:")
            for p in pesan_inconsistent_filtered:
                st.markdown(f"- {p}")
            st.caption("*Catatan: Model JST tetap dapat menganalisis data ini jika Anda mengklik tombol Analisis di bawah.*")

    # Tombol Analisis Sekarang (Selalu ditampilkan, dinonaktifkan jika kosong, invalid, atau total jam tidak logis)
    if st.button("🔍 Analisis Sekarang", use_container_width=True, disabled=is_disabled):
        # Jalankan analisis model JST
        data_baru = np.array(
            [[jam_tidur, mood, stres, jam_belajar, jam_hp, jumlah_tugas]],
            dtype=float
        )
        data_baru_clipped = np.clip(data_baru, repo.nilai_min, repo.nilai_max)
        # Prediksi Perceptron
        hasil_p = perceptron.predict(data_baru_clipped)[0]
        hasil_perceptron = "Produktif" if hasil_p == 1 else "Tidak Produktif"

        # Prediksi Kohonen (SOM)
        data_baru_norm = normalisasi_data_baru(data_baru_clipped, repo.nilai_min, repo.nilai_max)
        cluster = som.predict(data_baru_norm)[0]
        hasil_som = som.get_cluster_name(cluster)
        # Simpan ke session_state agar hasil tidak hilang saat berpindah tab
        st.session_state.last_analysis = {
            "inputs": [jam_tidur, mood, stres, jam_belajar, jam_hp, jumlah_tugas],
            "hasil_perceptron": hasil_perceptron,
            "hasil_som": hasil_som,
            "data_baru": data_baru,
            "data_baru_clipped": data_baru_clipped,
            "data_baru_norm": data_baru_norm,
            "cluster": cluster,
            "mood_val": mood if mood is not None else 0.0,
            "stres_val": stres if stres is not None else 0.0
        }

    # Render hasil analisis jika data input saat ini cocok dengan data yang terakhir dianalisis
    if not ada_yang_kosong and not ada_yang_invalid and not is_physically_impossible and "last_analysis" in st.session_state:
        la = st.session_state.last_analysis
        current_inputs = [jam_tidur, mood, stres, jam_belajar, jam_hp, jumlah_tugas]

        if la["inputs"] == current_inputs:
            hasil_perceptron = la["hasil_perceptron"]
            hasil_som = la["hasil_som"]
            data_baru = la["data_baru"]
            data_baru_clipped = la["data_baru_clipped"]
            data_baru_norm = la["data_baru_norm"]
            cluster = la["cluster"]
            mood_val = la["mood_val"]
            stres_val = la["stres_val"]

            st.divider()
            st.subheader("✅ Hasil Analisis")

            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.metric("Hasil Perceptron", hasil_perceptron)
            with col_h2:
                st.metric("Hasil Kohonen (Cluster)", hasil_som)

            st.info(get_saran(hasil_perceptron, hasil_som))

            st.write("### 📋 Data yang Dianalisis")
            st.dataframe(pd.DataFrame(data_baru, columns=repo.fitur), width="stretch")

            st.divider()
            st.subheader("🔍 Detail Matematis & Validasi Model")

            # ── 2 Tab: Perceptron | Kohonen ──────────────────────────────────
            tab_perceptron, tab_kohonen = st.tabs(["🧠 Bobot & Bias Perceptron", "🧩 Validasi Jarak Kohonen"])

            # ─────────────────────────── TAB PERCEPTRON ──────────────────────
            with tab_perceptron:
                # --- Ringkasan Training Epoch ---
                st.write("##### 📈 Ringkasan Training Model Perceptron")
                col_ep1, col_ep2, col_ep3 = st.columns(3)
                with col_ep1:
                    st.metric("Total Epoch", len(perceptron.errors))
                with col_ep2:
                    predictions_all = perceptron.predict(repo.X)
                    accuracy_all = (predictions_all == repo.y).mean() * 100
                    st.metric("Akurasi Training", f"{accuracy_all:.2f}%")
                with col_ep3:
                    st.metric("Bias Akhir", round(float(perceptron.bias), 4))

                fig_ep, ax_ep = plt.subplots(figsize=(8.5, 3.5))
                fig_ep.patch.set_facecolor('#0F172A')
                ax_ep.set_facecolor('#1E293B')
                epochs_range = range(1, len(perceptron.errors) + 1)
                ax_ep.plot(
                    epochs_range, perceptron.errors,
                    marker="o", color="#818CF8", linewidth=2.5,
                    markersize=6, markerfacecolor="#C084FC",
                    markeredgecolor="#0F172A", markeredgewidth=1.5
                )
                ax_ep.set_xlabel("Epoch", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
                ax_ep.set_ylabel("Total Error", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
                ax_ep.set_title("Kurva Penurunan Error Perceptron", fontsize=11, fontweight='bold', color='#F8FAFC', pad=12)
                ax_ep.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")
                ax_ep.spines['top'].set_visible(False)
                ax_ep.spines['right'].set_visible(False)
                ax_ep.spines['left'].set_color('#334155')
                ax_ep.spines['bottom'].set_color('#334155')
                ax_ep.tick_params(colors='#94A3B8', labelsize=9)
                st.pyplot(fig_ep)

                st.divider()
                st.write("##### Perhitungan Net Input Perceptron")
                bobot = perceptron.weights
                bias = perceptron.bias
                kontribusi = data_baru_clipped[0] * bobot
                total_net_input = np.sum(kontribusi) + bias

                df_perceptron = pd.DataFrame({
                    "Fitur": repo.fitur,
                    "Nilai Input (Clipped)": data_baru_clipped[0],
                    "Bobot Model": bobot,
                    "Kontribusi (Nilai x Bobot)": kontribusi
                })
                st.dataframe(df_perceptron, width="stretch")

                with st.expander("Bagaimana cara kerja bobot dan bias Perceptron?"):
                    st.write("Bobot dan bias di atas bukanlah angka yang ditebak secara acak, melainkan hasil dari **proses training (pembelajaran) iteratif** Perceptron menggunakan data latih sebelumnya.")
                    st.write("Setiap kali model melakukan kesalahan prediksi pada saat training, ia mengoreksi nilai bobot dan biasnya menggunakan rumus **Perceptron Learning Rule**:")
                    st.latex(r"\Delta w = \alpha \times (y_{\text{target}} - y_{\text{prediksi}}) \times x")
                    st.latex(r"w_{\text{baru}} = w_{\text{lama}} + \Delta w")
                    st.latex(r"b_{\text{baru}} = b_{\text{lama}} + \alpha \times (y_{\text{target}} - y_{\text{prediksi}})")

                    st.markdown(
                        """
                        **Keterangan:**
                        - $\\alpha$ (Alpha): **Learning Rate**, menentukan seberapa besar langkah koreksi.
                        - $y_{\\text{target}}$: Label asli dari dataset (1 untuk Produktif, 0 untuk Tidak Produktif).
                        - $y_{\\text{prediksi}}$: Prediksi model saat itu.
                        - $x$: Nilai fitur input.
                        **Contoh Logika:**
                        Jika model menebak "Produktif" (1) padahal aslinya "Tidak Produktif" (0), maka $(0 - 1) = -1$.
                        Bobot fitur yang bernilai positif akan **dikurangi** agar tebakan selanjutnya menjadi lebih akurat. Proses ini sudah diselesaikan pada tahap Training (lihat kurva di atas).
                        """
                    )

                st.markdown("**Persamaan Linear:**")
                st.latex(rf"\text{{Net Input}} = \sum (\text{{Nilai}} \times \text{{Bobot}}) + \text{{Bias}}")
                st.markdown(
                    f"""<div style="background: rgba(79,70,229,0.08); border: 1px solid rgba(79,70,229,0.25); 
                    border-radius: 10px; padding: 0.75rem 1rem; margin: 0.5rem 0;">
                    <span style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; 
                    letter-spacing: 0.05em; opacity: 0.7;">📌 Nilai Bias Model (tetap setelah training)</span><br/>
                    <span style="font-size: 1.2rem; font-weight: 800; color: #818CF8;">b = {bias:.4f}</span>
                    <span style="font-size: 0.82rem; opacity: 0.65; margin-left: 8px;">
                    (Ini adalah hasil training — bukan hasil prediksi)</span>
                    </div>""",
                    unsafe_allow_html=True
                )

                st.markdown("**Langkah Perhitungan Manual Perceptron:**")
                langkah_perkalian = " + ".join([f"({data_baru_clipped[0][i]:.1f} \\times {bobot[i]:.4f})" for i in range(len(bobot))])
                st.latex(rf"\text{{Net Input}} = {langkah_perkalian} + ({bias:.4f})")
                st.latex(rf"\text{{Net Input}} = {np.sum(kontribusi):.4f} + ({bias:.4f}) = {total_net_input:.4f}")

                # Hasil aktivasi dengan badge warna yang jelas
                status_aktivasi = ">= 0" if total_net_input >= 0 else "< 0"
                warna_aktivasi = "#10B981" if total_net_input >= 0 else "#EF4444"
                emoji_aktivasi = "✅" if total_net_input >= 0 else "❌"
                nilai_y = 1 if total_net_input >= 0 else 0
                st.markdown(
                    f"""<div style="background: rgba({('16,185,129' if total_net_input >= 0 else '239,68,68')},0.1); 
                    border: 1.5px solid {warna_aktivasi}; border-radius: 10px; 
                    padding: 0.85rem 1rem; margin: 0.75rem 0;">
                    <span style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; 
                    letter-spacing: 0.05em; color: {warna_aktivasi};">⚡ Hasil Fungsi Aktivasi Step (y = f(Net Input))</span><br/>
                    <span style="font-size: 1rem; font-weight: 600; opacity: 0.85;">
                    Net Input = <strong>{total_net_input:.4f}</strong> {status_aktivasi}
                    </span><br/>
                    <span style="font-size: 1.25rem; font-weight: 800; color: {warna_aktivasi};">
                    {emoji_aktivasi} y = {nilai_y} → <strong>{hasil_perceptron}</strong>
                    </span>
                    </div>""",
                    unsafe_allow_html=True
                )

                st.write("##### Visualisasi Garis Keputusan (Decision Boundary) Perceptron")
                fig_p, ax_p = plt.subplots(figsize=(8.5, 4.5))
                fig_p.patch.set_facecolor('#0F172A')
                ax_p.set_facecolor('#1E293B')
                colors_p = {0: "#F87171", 1: "#34D399"}
                for label_id in [0, 1]:
                    data_label = repo.df[repo.df["label"] == label_id]
                    ax_p.scatter(
                        data_label["mood"], data_label["stres"],
                        label="Produktif (Data Latih)" if label_id == 1 else "Tidak Produktif (Data Latih)",
                        color=colors_p[label_id], alpha=0.8, edgecolors="#1E293B", linewidths=0.7, s=80, zorder=3
                    )
                w = perceptron.weights
                b = perceptron.bias
                x_val = data_baru_clipped[0]
                const_val = b + w[0]*x_val[0] + w[3]*x_val[3] + w[4]*x_val[4] + w[5]*x_val[5]
                mood_range = np.linspace(repo.nilai_min[1], repo.nilai_max[1], 100)
                if w[2] != 0:
                    stres_range = - (w[1] * mood_range + const_val) / w[2]
                    ax_p.plot(mood_range, stres_range, color="#818CF8", linestyle="--", linewidth=2.5, label="Garis Pemisah Perceptron", zorder=4)
                elif w[1] != 0:
                    mood_c = - const_val / w[1]
                    ax_p.axvline(x=mood_c, color="#818CF8", linestyle="--", linewidth=2.5, label="Garis Pemisah Perceptron", zorder=4)
                ax_p.scatter(mood_val, stres_val, color="#FBBF24", edgecolors="black", s=260, marker="*", linewidths=1.2, label="Data Baru Anda", zorder=5)
                ax_p.set_xlabel("Mood Pagi", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
                ax_p.set_ylabel("Tingkat Stres", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
                ax_p.legend(frameon=True, facecolor='#1E293B', edgecolor='#334155', fontsize=9)
                ax_p.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")
                ax_p.spines['top'].set_visible(False)
                ax_p.spines['right'].set_visible(False)
                ax_p.spines['left'].set_color('#334155')
                ax_p.spines['bottom'].set_color('#334155')
                ax_p.tick_params(colors='#94A3B8', labelsize=9)
                ax_p.set_ylim(0, 11)
                st.pyplot(fig_p)
                st.write("*Catatan: Garis pemisah di atas digambar dengan mengasumsikan 4 fitur lainnya (Jam Tidur, Jam Belajar, Jam HP, Jumlah Tugas) bernilai konstan sesuai input baru Anda. Karena data latih lainnya memiliki nilai fitur pendukung yang bervariasi, beberapa titik data latih mungkin terlihat berada di sisi garis yang tidak sesuai dengan warna kelasnya.*")

            # ── 2 Tab: Kohonen ─────────────────────────
            with tab_kohonen:
                # Diagram Scatter Kohonen di bagian atas — visualisasi posisi data baru
                st.write("##### 📊 Posisi Data Baru pada Cluster Kohonen (Mood vs Stres)")
                fig, ax = plt.subplots(figsize=(8.5, 4.5))
                fig.patch.set_facecolor('#0F172A')
                ax.set_facecolor('#1E293B')
                colors = ["#F87171", "#60A5FA", "#34D399"]

                clusters_train = som.predict(repo.X_norm)
                df_cluster = repo.df.copy()
                df_cluster["cluster"] = clusters_train

                for idx, cluster_id in enumerate(np.unique(clusters_train)):
                    data_cluster = df_cluster[df_cluster["cluster"] == cluster_id]
                    ax.scatter(
                        data_cluster["mood"], data_cluster["stres"],
                        label=som.get_cluster_name(cluster_id),
                        color=colors[idx % len(colors)], alpha=0.8,
                        edgecolors="#1E293B", linewidths=0.7, s=80, zorder=3
                    )

                for idx, ww in enumerate(som.weights):
                    mood_c = ww[1] * (repo.nilai_max[1] - repo.nilai_min[1]) + repo.nilai_min[1]
                    stres_c = ww[2] * (repo.nilai_max[2] - repo.nilai_min[2]) + repo.nilai_min[2]
                    ax.scatter(
                        mood_c, stres_c,
                        color=colors[idx % len(colors)], edgecolors="#F8FAFC",
                        s=180, marker="X", linewidths=1.5,
                        label=f"Pusat Cluster: {som.get_cluster_name(idx)}", zorder=4
                    )

                ax.scatter(
                    mood_val, stres_val,
                    color="#FBBF24", edgecolors="black",
                    s=260, marker="*", linewidths=1.2,
                    label="Data Baru Anda ⭐", zorder=5
                )

                ax.set_xlabel("Mood Pagi", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
                ax.set_ylabel("Tingkat Stres", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
                ax.set_title("Posisi Data Baru pada Cluster Kohonen (SOM)", fontsize=11, fontweight='bold', color='#F8FAFC', pad=12)
                ax.legend(frameon=True, facecolor='#1E293B', edgecolor='#334155', fontsize=9)
                ax.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#334155')
                ax.spines['bottom'].set_color('#334155')
                ax.tick_params(colors='#94A3B8', labelsize=9)
                st.pyplot(fig)

                st.divider()

                # Tabel validasi jarak Euclidean
                st.write("##### 📐 Jarak Euclidean ke Centroid Kohonen (SOM)")
                distances = np.linalg.norm(som.weights - data_baru_norm, axis=1)
                df_som_val = pd.DataFrame({
                    "Nama Cluster": [som.get_cluster_name(i) for i in range(len(som.weights))],
                    "Jarak Euclidean ke Centroid": distances,
                    "Status": ["Pemenang (Terdekat) 🏆" if i == cluster else "Bukan Pemenang" for i in range(len(som.weights))]
                })
                st.dataframe(df_som_val, width="stretch")
                st.write("*Catatan: Kohonen (SOM) mengklasifikasikan input ke cluster dengan jarak Euclidean terkecil.*")

                st.write("##### Langkah Perhitungan Manual Jarak Euclidean (Kohonen SOM)")
                st.markdown("Rumus: $d = \\sqrt{\\sum_{j=1}^6 (x_j - w_j)^2}$")

                x_norm = data_baru_norm[0]
                for i in range(len(som.weights)):
                    cluster_name = som.get_cluster_name(i)
                    w_c = som.weights[i]
                    selisih_kuadrat = [(x_norm[j] - w_c[j])**2 for j in range(6)]
                    langkah_sum = " + ".join([f"({x_norm[j]:.4f} - {w_c[j]:.4f})^2" for j in range(6)])
                    langkah_num = " + ".join([f"{s:.4f}" for s in selisih_kuadrat])
                    total_sum = sum(selisih_kuadrat)
                    with st.expander(f"🔍 Langkah Perhitungan ke Cluster {cluster_name}"):
                        st.markdown(f"**Jarak ke Centroid {cluster_name}:**")
                        st.latex(rf"d_{{\text{{{cluster_name}}}}} = \sqrt{{{langkah_sum}}}")
                        st.latex(rf"d_{{\text{{{cluster_name}}}}} = \sqrt{{{langkah_num}}}")
                        st.latex(rf"d_{{\text{{{cluster_name}}}}} = \sqrt{{{total_sum:.4f}}} = {np.sqrt(total_sum):.4f}")

                st.write("##### Bobot Centroid (Pusat Koordinat) Masing-Masing Cluster Kohonen (SOM)")
                bobot_som = pd.DataFrame(som.weights, columns=repo.fitur)
                bobot_som.insert(0, "Cluster", [som.get_cluster_name(i) for i in range(len(som.weights))])
                st.dataframe(bobot_som, width="stretch")
