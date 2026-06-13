import numpy as np


def normalisasi(X):
    """Normalisasi data ke rentang [0, 1] menggunakan Min-Max Scaling."""
    X = np.array(X, dtype=float)
    nilai_min = X.min(axis=0)
    nilai_max = X.max(axis=0)
    return (X - nilai_min) / (nilai_max - nilai_min), nilai_min, nilai_max


def normalisasi_data_baru(data_baru, nilai_min, nilai_max):
    """Normalisasi data input baru menggunakan parameter min/max dari data latih dengan batasan [0, 1]."""
    norm_data = (data_baru - nilai_min) / (nilai_max - nilai_min)
    return np.clip(norm_data, 0.0, 1.0)


def get_saran(hasil_perceptron, hasil_som):
    """Memberikan saran aktivitas berdasarkan hasil Perceptron dan SOM."""
    if hasil_som == "Produktif" and hasil_perceptron == "Produktif":
        return "Kondisi kamu cukup baik. Pertahankan pola tidur, mood, dan jam belajar yang stabil."
    elif hasil_som == "Burnout":
        return "Kamu terindikasi burnout. Kurangi penggunaan HP, istirahat cukup, dan prioritaskan tugas yang paling penting."
    elif hasil_som == "Santai":
        return "Kondisi kamu cukup santai. Coba tingkatkan jam belajar sedikit agar produktivitas lebih optimal."
    else:
        return "Kondisi kamu kurang produktif. Perbaiki jam tidur, kurangi stres, dan atur ulang waktu belajar."
