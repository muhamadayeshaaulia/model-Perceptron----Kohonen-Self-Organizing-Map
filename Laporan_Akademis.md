# LAPORAN AKADEMIS: Deteksi Produktivitas dan Kejenuhan (Burnout) Mahasiswa

## BAB I: Latar Belakang
Mahasiswa sering kali dihadapkan pada rutinitas padat yang mencakup jam belajar, pengerjaan tugas, hingga aktivitas hiburan seperti bermain *smartphone*. Manajemen waktu dan emosi yang buruk (seperti kurang tidur dan tingginya tingkat stres) dapat berdampak langsung pada produktivitas, bahkan mengarah pada *burnout* (kejenuhan ekstrem). 
Masalah ini sangat penting untuk diselesaikan karena deteksi dini terhadap pola aktivitas harian dapat membantu mahasiswa mengevaluasi keseimbangan hidup mereka (*work-life balance*). Dengan menggunakan Jaringan Saraf Tiruan (JST), kita dapat menemukan pola tersembunyi dari rutinitas harian dan memprediksi secara otomatis apakah aktivitas pada hari tersebut tergolong produktif atau tidak, serta mengelompokkan kondisi psikologis mahasiswa untuk mencegah terjadinya *burnout*.

---

## BAB II: Arsitektur Model
Penelitian ini menggunakan dua pendekatan Jaringan Saraf Tiruan (JST), yaitu untuk *Supervised Learning* (Klasifikasi) dan *Unsupervised Learning* (Clustering).

### 1. Model Perceptron (Klasifikasi Produktivitas)
Model ini menggunakan arsitektur **Single Layer Perceptron** karena masalah yang diselesaikan bersifat *linearly separable* (dapat dipisahkan dengan garis lurus).
- **Jumlah Input Layer**: 6 Node (Mewakili 6 fitur: Jam Tidur, Mood Pagi, Tingkat Stres, Jam Belajar, Jam HP, Jumlah Tugas).
- **Jumlah Hidden Layer**: 0 (Tidak ada *hidden layer* pada Single Layer Perceptron).
- **Jumlah Output Layer**: 1 Node (Menghasilkan nilai biner: 1 untuk Produktif, 0 untuk Tidak Produktif).
- **Fungsi Aktivasi**: **Step Function (Fungsi Undak Biner)**. Jika hasil kombinasi linear (Net Input) $\ge 0$, maka output = 1. Jika $< 0$, maka output = 0.

### 2. Model Kohonen Self-Organizing Map / SOM (Clustering Kondisi)
- **Jumlah Input Layer**: 6 Node (Mewakili 6 fitur ternormalisasi).
- **Jumlah Hidden Layer**: 0 (SOM menggunakan *competitive layer* yang bertindak langsung sebagai output).
- **Jumlah Output Layer**: 3 Node (Mewakili 3 klaster/centroid: Produktif, Santai, Burnout).
- **Fungsi Aktivasi / Metode Utama**: **Winner-Takes-All (Jarak Euclidean)**. Node output dengan jarak Euclidean terdekat dengan data input akan menjadi "Pemenang" dan bobotnya akan diperbarui.

---

## BAB III: Perhitungan Matematis
Berikut adalah simulasi perhitungan **Perceptron Learning Rule** (1 Iterasi Forward & Backward).
Misalkan model diinisialisasi dengan bobot $w = [0, 0, 0, 0, 0, 0]$, Bias $b = 0$, dan *Learning Rate* ($\alpha$) = 0.1.

Diambil satu sampel data mahasiswa yang sedang tidak produktif (Target = 0):
- Input ($x$): Jam Tidur=5, Mood=6, Stres=6, Belajar=3, HP=5, Tugas=4.

### 1. Proses Forward (Propagasi Maju)
Menghitung Net Input ($y_{in}$):
$$y_{in} = \sum (x_i \cdot w_i) + b$$
$$y_{in} = (5\cdot0) + (6\cdot0) + (6\cdot0) + (3\cdot0) + (5\cdot0) + (4\cdot0) + 0 = 0$$

Menggunakan Fungsi Aktivasi Step:
Karena $y_{in} \ge 0$ ($0 \ge 0$), maka Prediksi Output ($y_{prediksi}$) = **1**.
Ternyata tebakan model salah, karena Target aslinya adalah **0**.

### 2. Proses Backward (Pembaruan Bobot)
Karena terjadi *error* ($Target - Prediksi = 0 - 1 = -1$), model memperbarui bobotnya menggunakan rumus:
$$w_{baru} = w_{lama} + \alpha \times \text{Error} \times x$$
$$b_{baru} = b_{lama} + \alpha \times \text{Error}$$

Perhitungan Bobot Baru:
- $w_{\text{tidur}} = 0 + (0.1 \times -1 \times 5) = -0.5$
- $w_{\text{mood}} = 0 + (0.1 \times -1 \times 6) = -0.6$
- $w_{\text{stres}} = 0 + (0.1 \times -1 \times 6) = -0.6$
- $w_{\text{belajar}} = 0 + (0.1 \times -1 \times 3) = -0.3$
- $w_{\text{hp}} = 0 + (0.1 \times -1 \times 5) = -0.5$
- $w_{\text{tugas}} = 0 + (0.1 \times -1 \times 4) = -0.4$
- $b_{baru} = 0 + (0.1 \times -1) = -0.1$

Bobot inilah yang akan digunakan untuk memproses baris data selanjutnya.

---

## BAB IV: Analisis Hasil
Berdasarkan uji coba training model Perceptron pada aplikasi, perubahan pada nilai *Learning Rate* ($\eta$ atau $\alpha$) dan jumlah *Epoch* memberikan dampak signifikan terhadap akurasi dan konvergensi:

1. **Pengaruh Jumlah Epoch**: 
   Epoch menentukan berapa kali model membaca ulang seluruh dataset. Jika jumlah epoch terlalu kecil (misal: 2), model belum sempat memperbaiki bobot secara maksimal, sehingga akurasinya rendah (sering terjadi *underfitting*). Saat epoch dinaikkan, tingkat total error akan menurun secara bertahap hingga mencapai 0 (konvergen).
2. **Pengaruh Learning Rate ($\eta$)**:
   - Jika *learning rate* terlalu **besar** (misal 0.9), model memperbarui bobot terlalu agresif. Hal ini menyebabkan kurva error melonjak naik-turun secara ekstrem (osilasi) dan bisa jadi gagal mencapai titik konvergen (error = 0) karena melompati titik optimal.
   - Jika *learning rate* terlalu **kecil** (misal 0.01), langkah pembaruan bobot sangat lambat. Akurasi tetap bisa mencapai 100%, namun membutuhkan jumlah *epoch* yang jauh lebih banyak dan waktu komputasi yang lebih lama.
   - Nilai moderat seperti $\alpha = 0.1$ terbukti paling optimal untuk menyeimbangkan kecepatan dan stabilitas training pada dataset ini.

---

## BAB V: Kesimpulan
Model Jaringan Saraf Tiruan yang dibangun telah **berhasil menyelesaikan masalah**. 
1. **Perceptron** terbukti mampu mempelajari bobot dari masing-masing parameter (seperti memberikan penalti bobot negatif untuk stres dan jam HP) untuk mengklasifikasikan hari mahasiswa ke dalam status **Produktif** atau **Tidak Produktif** dengan akurasi yang tinggi.
2. **Kohonen SOM** berhasil memetakan dataset tanpa label ke dalam peta persepsi yang memisahkan profil mahasiswa secara logis ke dalam 3 kondisi (*Produktif, Santai, Burnout*). 

Dengan demikian, sistem deteksi ini dapat diandalkan sebagai alat bantu evaluasi psikologis dan akademis bagi mahasiswa untuk mengatur rutinitas hariannya secara lebih sehat.
