import numpy as np
from numpy.typing import NDArray


class KohonenSOM:
    def __init__(self, jumlah_cluster=3, learning_rate=0.5, epoch=100):
        self.jumlah_cluster = jumlah_cluster
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weights: NDArray[np.float64] = np.empty((0, 0))
        self.cluster_names = {}

    def fit(self, X):
        jumlah_fitur = X.shape[1]

        # Inisialisasi bobot random untuk setiap cluster
        np.random.seed(42)
        self.weights = np.random.rand(self.jumlah_cluster, jumlah_fitur)

        for _ in range(self.epoch):
            for data in X:
                # Hitung jarak Euclidean antara data dan bobot cluster
                distances = np.linalg.norm(self.weights - data, axis=1)

                # Cari cluster pemenang
                winner = np.argmin(distances)

                # Update bobot cluster pemenang
                self.weights[winner] += self.learning_rate * (data - self.weights[winner])

        # Setelah training, beri nama cluster otomatis
        self.set_cluster_names()

    def set_cluster_names(self):
        """
        Urutan fitur:
        0 = jam_tidur
        1 = mood
        2 = stres
        3 = jam_belajar
        4 = jam_hp
        5 = jumlah_tugas

        Skor produktivitas tinggi jika:
        jam_tidur tinggi, mood tinggi, jam_belajar tinggi,
        stres rendah, jam_hp rendah, jumlah_tugas rendah.
        """

        scores = []

        for i, w in enumerate(self.weights):
            score = (
                w[0] +      # jam tidur
                w[1] +      # mood
                w[3] -      # jam belajar
                w[2] -      # stres
                w[4] -      # jam HP
                w[5]        # jumlah tugas
            )
            scores.append((i, score))

        # Urutkan dari skor paling rendah sampai tinggi
        sorted_scores = sorted(scores, key=lambda x: x[1])

        # Skor terendah = Burnout
        # Skor tengah = Santai
        # Skor tertinggi = Produktif
        self.cluster_names[sorted_scores[0][0]] = "Burnout"
        self.cluster_names[sorted_scores[1][0]] = "Santai"
        self.cluster_names[sorted_scores[2][0]] = "Produktif"

    def predict_one(self, x):
        distances = np.linalg.norm(self.weights - x, axis=1)
        cluster = np.argmin(distances)
        return cluster

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def get_cluster_name(self, cluster):
        return self.cluster_names.get(cluster, "Tidak diketahui")