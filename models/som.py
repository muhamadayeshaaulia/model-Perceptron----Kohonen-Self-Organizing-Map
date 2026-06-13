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
        # Inisialisasi bobot menggunakan sampel data latih asli (mencegah dead neuron)
        np.random.seed(42)
        indices = np.random.choice(len(X), self.jumlah_cluster, replace=False)
        self.weights = X[indices].copy()

        for _ in range(self.epoch):
            for data in X:
                distances = np.linalg.norm(self.weights - data, axis=1)
                winner = np.argmin(distances)
                self.weights[winner] += self.learning_rate * (data - self.weights[winner])

        self.set_cluster_names()

    def set_cluster_names(self):
        """
        Menentukan label klaster secara dinamis dan akurat:
        1. Klaster dengan bobot stres (indeks 2) tertinggi didefinisikan sebagai Burnout.
        2. Dari 2 klaster tersisa, klaster dengan bobot jam belajar (indeks 3) tertinggi
           didefinisikan sebagai Produktif, sedangkan yang lainnya sebagai Santai.
        """
        # Cari klaster Burnout (bobot stres tertinggi)
        stres_weights = [w[2] for w in self.weights]
        burnout_idx = int(np.argmax(stres_weights))
        
        # Dapatkan indeks 2 klaster lainnya
        remaining_indices = [i for i in range(self.jumlah_cluster) if i != burnout_idx]
        idx_a, idx_b = remaining_indices
        
        # Bandingkan bobot jam belajar (indeks 3)
        if self.weights[idx_a][3] > self.weights[idx_b][3]:
            produktif_idx = idx_a
            santai_idx = idx_b
        else:
            produktif_idx = idx_b
            santai_idx = idx_a
            
        self.cluster_names[burnout_idx] = "Burnout"
        self.cluster_names[santai_idx] = "Santai"
        self.cluster_names[produktif_idx] = "Produktif"

    def predict_one(self, x):
        distances = np.linalg.norm(self.weights - x, axis=1)
        return int(np.argmin(distances))

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def get_cluster_name(self, cluster):
        return self.cluster_names.get(cluster, "Tidak diketahui")
