import pandas as pd
from perceptron import Perceptron

# Baca dataset
df = pd.read_csv("dataset.csv")

# Ambil fitur input
X = df[["jam_tidur", "mood", "stres", "jam_belajar", "jam_hp", "jumlah_tugas"]].values

# Ambil label
y = df["label"].values

# Buat dan latih model
model = Perceptron(learning_rate=0.1, epoch=50)
model.fit(X, y)

# Prediksi data contoh
data_baru = [[7, 8, 3, 5, 2, 2]]
hasil = model.predict(data_baru)

print("Bobot akhir:", model.weights)
print("Bias akhir:", model.bias)
print("Error tiap epoch:", model.errors)

if hasil[0] == 1:
    print("Hasil Prediksi: Produktif")
else:
    print("Hasil Prediksi: Tidak Produktif")
