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


def cek_konsistensi(jam_tidur, mood, stres, jam_belajar, jam_hp, jumlah_tugas):
    """
    Mengecek konsistensi data input berdasarkan logika aktivitas harian dan pola dataset.
    Mengembalikan list berisi string pesan ketidakkonsistenan. Jika kosong, berarti data konsisten.
    """
    pesan = []
    
    # 1. Total jam melebihi 24 jam
    if jam_tidur + jam_belajar + jam_hp > 24:
        pesan.append(
            f"Jumlah **Jam Tidur** ({jam_tidur} jam), **Jam Belajar** ({jam_belajar} jam), dan **Jam HP** ({jam_hp} jam) "
            f"berjumlah **{jam_tidur + jam_belajar + jam_hp} jam**, melebihi total 24 jam dalam sehari. Ini tidak mungkin secara fisik."
        )
        
    # 2. Mood dan Stres kontradiktif (keduanya tinggi)
    if mood >= 7 and stres >= 7:
        pesan.append(
            f"**Mood Pagi** ({mood}) dan **Tingkat Stres** ({stres}) keduanya bernilai tinggi. "
            "Secara psikologis, mood yang sangat baik biasanya tidak terjadi bersamaan dengan stres yang sangat tinggi."
        )
        
    # 3. Mood dan Stres kontradiktif (keduanya rendah)
    if mood <= 4 and stres <= 3:
        pesan.append(
            f"**Mood Pagi** ({mood}) dan **Tingkat Stres** ({stres}) keduanya bernilai rendah. "
            "Biasanya, jika mood pagi buruk, tingkat stres cenderung lebih tinggi."
        )
        
    # 4. Jam Belajar dan Jam HP keduanya sangat tinggi
    if jam_belajar >= 5 and jam_hp >= 7:
        pesan.append(
            f"**Jam Belajar** ({jam_belajar} jam) dan **Jam HP** ({jam_hp} jam) keduanya bernilai tinggi. "
            "Kombinasi ini kontradiktif karena memakan waktu produktif dan non-produktif yang maksimal secara bersamaan."
        )
        
    # 5. Kurang tidur ekstrem tapi mood sangat tinggi
    if jam_tidur <= 4 and mood >= 8:
        pesan.append(
            f"**Jam Tidur** sangat kurang ({jam_tidur} jam) tetapi **Mood Pagi** Anda sangat tinggi ({mood}). "
            "Kurang tidur biasanya berdampak negatif pada suasana hati di pagi hari."
        )
        
    return pesan

