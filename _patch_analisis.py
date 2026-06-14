"""Patch script: tambahkan ringkasan epoch ke analisis_view.py"""
path = 'views/analisis_view.py'

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Cari baris "##### Perhitungan Net Input Perceptron"
target = '            st.write("##### Perhitungan Net Input Perceptron")\n'
insert_idx = None
for i, line in enumerate(lines):
    if 'Perhitungan Net Input Perceptron' in line:
        insert_idx = i
        break

if insert_idx is None:
    print("TIDAK DITEMUKAN target line. Isi baris sekitar tab_perceptron:")
    for i, line in enumerate(lines):
        if 'tab_perceptron' in line:
            print(f"  [{i}]: {repr(line)}")
else:
    print(f"Ditemukan di baris {insert_idx}: {repr(lines[insert_idx])}")
    epoch_block = [
        '            # --- Ringkasan Training Epoch ---\n',
        '            st.write("##### \U0001f4c8 Ringkasan Training Model Perceptron")\n',
        '            col_ep1, col_ep2, col_ep3 = st.columns(3)\n',
        '            with col_ep1:\n',
        '                st.metric("Total Epoch", len(perceptron.errors))\n',
        '            with col_ep2:\n',
        '                predictions_all = perceptron.predict(repo.X)\n',
        '                accuracy_all = (predictions_all == repo.y).mean() * 100\n',
        '                st.metric("Akurasi Training", f"{accuracy_all:.2f}%")\n',
        '            with col_ep3:\n',
        '                st.metric("Bias Akhir", round(float(perceptron.bias), 4))\n',
        '\n',
        '            fig_ep, ax_ep = plt.subplots(figsize=(8.5, 3.5))\n',
        '            fig_ep.patch.set_facecolor("#0F172A")\n',
        '            ax_ep.set_facecolor("#1E293B")\n',
        '            epochs_range = range(1, len(perceptron.errors) + 1)\n',
        '            ax_ep.plot(\n',
        '                epochs_range, perceptron.errors,\n',
        '                marker="o", color="#818CF8", linewidth=2.5,\n',
        '                markersize=6, markerfacecolor="#C084FC",\n',
        '                markeredgecolor="#0F172A", markeredgewidth=1.5\n',
        '            )\n',
        '            ax_ep.set_xlabel("Epoch", fontsize=10, fontweight="semibold", color="#94A3B8", labelpad=8)\n',
        '            ax_ep.set_ylabel("Total Error", fontsize=10, fontweight="semibold", color="#94A3B8", labelpad=8)\n',
        '            ax_ep.set_title("Kurva Penurunan Error Perceptron", fontsize=11, fontweight="bold", color="#F8FAFC", pad=12)\n',
        '            ax_ep.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")\n',
        '            ax_ep.spines["top"].set_visible(False)\n',
        '            ax_ep.spines["right"].set_visible(False)\n',
        '            ax_ep.spines["left"].set_color("#334155")\n',
        '            ax_ep.spines["bottom"].set_color("#334155")\n',
        '            ax_ep.tick_params(colors="#94A3B8", labelsize=9)\n',
        '            st.pyplot(fig_ep)\n',
        '\n',
        '            st.divider()\n',
        '            st.write("##### Perhitungan Net Input Perceptron")\n',
    ]
    # Ganti baris target dengan epoch block (target diganti oleh baris terakhir epoch_block)
    new_lines = lines[:insert_idx] + epoch_block + lines[insert_idx + 1:]
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"SUCCESS: Epoch block ditambahkan sebelum baris {insert_idx}")
