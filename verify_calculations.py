import numpy as np
import pandas as pd
from repositories.dataset_repository import DatasetRepository
from services.model_service import load_model
from utils.helper import normalisasi_data_baru

def run_verification():
    print("=== MEMULAI VERIFIKASI MATEMATIS SISTEM ===")
    
    # 1. Load data & model
    repo, perceptron, som = load_model()
    print("✔ Model Perceptron & Kohonen (SOM) berhasil di-load.")
    print(f"✔ Dataset loaded. Jumlah data: {len(repo.df)}")
    
    # Test cases: beberapa skenario data harian mahasiswa
    test_cases = [
        # [jam_tidur, mood, stres, jam_belajar, jam_hp, jumlah_tugas]
        {"name": "Skenario Produktif", "data": [7.0, 8.0, 2.0, 5.0, 2.0, 1.0]},
        {"name": "Skenario Burnout", "data": [4.0, 4.0, 9.0, 1.0, 8.0, 6.0]},
        {"name": "Skenario Santai", "data": [6.0, 6.0, 5.0, 3.0, 5.0, 3.0]}
    ]
    
    for case in test_cases:
        name = case["name"]
        raw_input = np.array([case["data"]], dtype=float)
        
        print(f"\n--- Menguji Skenario: {name} ---")
        print(f"Input Riil: {case['data']}")
        
        # --- VERIFIKASI PERCEPTRON ---
        # 1. Prediksi asli model
        pred_model_p = perceptron.predict(raw_input)[0]
        
        # 2. Perhitungan manual
        bobot = perceptron.weights
        bias = perceptron.bias
        kontribusi = raw_input[0] * bobot
        net_input_manual = np.sum(kontribusi) + bias
        pred_manual_p = 1 if net_input_manual >= 0 else 0
        
        status_p = "Produktif" if pred_manual_p == 1 else "Tidak Produktif"
        print(f"[Perceptron] Perhitungan Manual Net Input: {net_input_manual:.4f}")
        print(f"[Perceptron] Hasil Aktivasi Manual: {pred_manual_p} ({status_p})")
        print(f"[Perceptron] Hasil Prediksi Model: {pred_model_p}")
        
        # Cek apakah hasil perhitungan manual sama dengan prediksi model
        assert pred_manual_p == pred_model_p, f"❌ MISMATCH pada Perceptron untuk {name}!"
        print("✔ Verifikasi Perceptron: Lolos (Perhitungan manual cocok dengan prediksi model)")
        
        # --- VERIFIKASI KOHONEN (SOM) ---
        # Normalisasi input
        norm_input = normalisasi_data_baru(raw_input, repo.nilai_min, repo.nilai_max)
        
        # 1. Prediksi asli model
        pred_model_som = som.predict(norm_input)[0]
        
        # 2. Perhitungan manual Jarak Euclidean ke setiap centroid
        distances_manual = []
        for i, w_c in enumerate(som.weights):
            dist = np.sqrt(np.sum((norm_input[0] - w_c) ** 2))
            distances_manual.append(dist)
            print(f"[Kohonen] Jarak ke Centroid {som.get_cluster_name(i)}: {dist:.4f}")
            
        pred_manual_som = np.argmin(distances_manual)
        print(f"[Kohonen] Cluster Terdekat (Manual): Cluster {pred_manual_som} ({som.get_cluster_name(pred_manual_som)})")
        print(f"[Kohonen] Cluster Terdekat (Model): Cluster {pred_model_som} ({som.get_cluster_name(pred_model_som)})")
        
        # Cek apakah hasil perhitungan manual sama dengan prediksi model
        assert pred_manual_som == pred_model_som, f"❌ MISMATCH pada Kohonen SOM untuk {name}!"
        print("✔ Verifikasi Kohonen (SOM): Lolos (Perhitungan manual cocok dengan hasil clustering model)")
        
    print("\n=======================================================")
    print("🎉 SEMUA VERIFIKASI SELESAI & LOLOS 100%! 🎉")
    print("Perhitungan manual yang tertera di UI aplikasi Anda terbukti")
    print("akurat dan konsisten dengan output model JST di backend.")
    print("=======================================================")

if __name__ == "__main__":
    run_verification()
