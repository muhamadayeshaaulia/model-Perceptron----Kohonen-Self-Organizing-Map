# 🧠 Aplikasi Deteksi Produktivitas & Burnout Mahasiswa

Aplikasi berbasis web ini dirancang untuk membantu mahasiswa memantau kondisi produktivitas harian dan mendeteksi potensi *burnout* (kejenuhan). Aplikasi ini dibangun menggunakan antarmuka interaktif **Streamlit** dan memanfaatkan dua buah model Jaringan Saraf Tiruan (JST) untuk melakukan analisis:
1. **Perceptron** (Supervised Learning - Klasifikasi)
2. **Kohonen Self-Organizing Map / SOM** (Unsupervised Learning - Clustering)

---

## 🎯 Hasil Analisis Model (Outcomes)

Aplikasi ini menggunakan 6 parameter input harian: **Jam Tidur, Mood Pagi (1-10), Tingkat Stres (1-10), Jam Belajar, Jam Bermain HP, dan Jumlah Tugas**.

Dari input tersebut, model akan memberikan hasil sebagai berikut:

### 1. Klasifikasi Perceptron (2 Kategori)
Model Perceptron dilatih menggunakan dataset yang sudah memiliki label untuk mengklasifikasikan kondisi harian mahasiswa ke dalam **2 status tegas**:
- ✅ **Produktif (Label 1)**: Kondisi optimal di mana jam tidur cukup, stres terkendali, dan waktu belajar efektif.
- ❌ **Tidak Produktif (Label 0)**: Kondisi kurang optimal, biasanya ditandai dengan stres tinggi, kurang tidur, dan terlalu banyak bermain HP.

### 2. Clustering Kohonen SOM (3 Klaster)
Berbeda dengan Perceptron, Kohonen SOM tidak menggunakan label saat berlatih. Ia mengelompokkan mahasiswa berdasarkan kemiripan pola keseharian (menggunakan jarak Euclidean) ke dalam **3 klaster kondisi psikologis**:
- 🌟 **Klaster Produktif**: Keseimbangan yang baik antara beban tugas, waktu belajar, dan manajemen stres.
- ☕ **Klaster Santai**: Cenderung memiliki sedikit tugas, stres rendah, namun waktu belajar lebih sedikit dan jam HP lebih tinggi.
- ⚠️ **Klaster Burnout**: Kondisi kritis yang ditandai dengan tingkat stres maksimal, jam tidur minim, dan beban tugas yang sangat tinggi. Membutuhkan istirahat segera.

---

## 🛠️ Library yang Digunakan

Aplikasi ini dibangun menggunakan murni Python dengan beberapa library utama berikut:

- **[`streamlit`](https://streamlit.io/)**: Digunakan sebagai framework utama pembuat antarmuka web (UI) interaktif, navigasi sidebar, dan komponen visual *glassmorphism* tanpa perlu menulis HTML/CSS dari nol (meskipun dipercantik dengan kustomisasi CSS tambahan).
- **[`pandas`](https://pandas.pydata.org/)**: Digunakan untuk manipulasi dan analisis data tabular (DataFrames), termasuk membaca dataset dari file CSV, menata tabel hasil metrik, dan menyiapkan data fitur.
- **[`numpy`](https://numpy.org/)**: Digunakan sebagai mesin komputasi matriks dan matematika tingkat tinggi. NumPy menangani operasi matriks seperti perhitungan *dot product* bobot Perceptron, perhitungan *Euclidean distance* untuk SOM, dan normalisasi Min-Max.
- **[`matplotlib`](https://matplotlib.org/)**: Digunakan untuk membuat visualisasi data saintifik seperti plot sebaran klaster (*scatter plot*), kurva penurunan error (*epoch*), dan garis keputusan matematis (*Decision Boundary*).

---

## 🚀 Cara Menjalankan Aplikasi Lokal

Jika Anda ingin menjalankan atau mengembangkan aplikasi ini di komputer lokal, ikuti langkah berikut:

### 1. Persiapan Virtual Environment
Disarankan menggunakan *virtual environment* agar dependensi tidak bentrok dengan proyek Python lain.
```bash
python3 -m venv venv
source venv/bin/activate  # Untuk Linux/Mac
# venv\Scripts\activate   # Untuk Windows
```

### 2. Install Library
Install semua library yang dibutuhkan dengan perintah berikut:
```bash
pip install numpy pandas matplotlib streamlit
```

### 3. Jalankan Streamlit
Setelah instalasi selesai, jalankan server Streamlit lokal:
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

*Dikembangkan untuk eksperimen Jaringan Saraf Tiruan (JST) - Perceptron & Kohonen SOM.*