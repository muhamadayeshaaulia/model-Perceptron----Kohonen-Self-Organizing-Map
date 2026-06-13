import pandas as pd
import numpy as np
from som import KohonenSOM


def normalisasi(X):
    X = np.array(X, dtype=float)
    nilai_min = X.min(axis=0)
    nilai_max = X.max(axis=0)
    return (X - nilai_min) / (nilai_max - nilai_min)


# Baca dataset
df = pd.read_csv("dataset.csv")

# Ambil fitur input tanpa label
X = df[["jam_tidur", "mood", "stres", "jam_belajar", "jam_hp", "jumlah_tugas"]].values

# Normalisasi data agar rentangnya 0 sampai 1
X_norm = normalisasi(X)

# Buat dan latih model SOM
som = KohonenSOM(jumlah_cluster=3, learning_rate=0.5, epoch=100)
som.fit(X_norm)

# Prediksi semua data
clusters = som.predict(X_norm)

print("Bobot akhir SOM:")
print(som.weights)

print("\nHasil cluster setiap data:")
for i, cluster in enumerate(clusters):
    print(f"Data ke-{i+1}: Cluster {cluster} - {som.get_cluster_name(cluster)}")

# Contoh data baru
data_baru = np.array([[7, 8, 3, 5, 2, 2]], dtype=float)

# Normalisasi data baru pakai min dan max dataset
nilai_min = X.min(axis=0)
nilai_max = X.max(axis=0)
data_baru_norm = (data_baru - nilai_min) / (nilai_max - nilai_min)

hasil_cluster = som.predict(data_baru_norm)[0]

print("\nData baru:")
print("Cluster:", hasil_cluster)
print("Kategori:", som.get_cluster_name(hasil_cluster))
