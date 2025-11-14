import pandas as pd
import joblib

# load model
model = joblib.load("model_gula.pkl")

print("===== Prediksi Gula Darah =====")

# input dari user
sarapan = int(input("Apakah Anda sarapan? (1=Ya, 0=Tidak): "))
berat = float(input("Masukkan berat badan Anda (kg): "))
aktivitas = float(input("Berapa jam aktivitas/olahraga hari ini?: "))

# buat dataframe
user = pd.DataFrame({
    "Sarapan": [sarapan],
    "Berat_Badan": [berat],
    "Aktivitas": [aktivitas]
})

# prediksi
hasil = model.predict(user)[0]

# probabilitas
probs = model.predict_proba(user)[0]
kelas = model.classes_

print("\n===== HASIL PREDIKSI =====")
print(f"Gula darah Anda diprediksi: **{hasil}**\n")

print("Probabilitas masing-masing kategori:")
for k, p in zip(kelas, probs):
    print(f"  {k}: {p*100:.2f}%")

# rekomendasi kesehatan sederhana
print("\n===== SARAN KESEHATAN =====")
if hasil == "Tinggi":
    print("- Kurangi makanan manis\n- Banyak minum air putih\n- Perbanyak aktivitas ringan")
elif hasil == "Rendah":
    print("- Makan makanan berkarbohidrat\n- Jangan telat makan\n- Boleh konsumsi minuman manis sedikit")
else:
    print("- Pertahankan pola hidup sehat!\n- Sarapan teratur dan aktivitas cukup")
