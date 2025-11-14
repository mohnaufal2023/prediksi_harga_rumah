import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

# Generate data
sarapan = np.random.choice([0,1], size=n, p=[0.4, 0.6])
berat = np.random.normal(70, 12, n).astype(int)
aktivitas = np.random.randint(0, 10, n)
usia = np.random.randint(15, 70, n)
tidur = np.random.randint(3, 10, n)
gula_harian = np.random.randint(0, 150, n)
riwayat = np.random.choice([0,1], size=n, p=[0.7, 0.3])

# Skor medis
score = (
    (sarapan == 0) * 15 +
    (berat - 70) * 0.6 +
    (10 - aktivitas) * 1.2 +
    (usia - 40) * 0.5 +
    (7 - tidur) * 2.5 +
    gula_harian * 0.2 +
    riwayat * 20
)

# Label berdasarkan score
label = []
for s in score:
    if s < 20:
        label.append("Rendah")
    elif s < 55:
        label.append("Normal")
    else:
        label.append("Tinggi")

df = pd.DataFrame({
    'Sarapan': sarapan,
    'Berat_Badan': berat,
    'Aktivitas': aktivitas,
    'Usia': usia,
    'Tidur': tidur,
    'Gula_Harian': gula_harian,
    'Riwayat_Keluarga': riwayat,
    'Gula_Darah': label
})

df.to_csv("gula_dataset_1000.csv", index=False)
print("Dataset 1000 baris berhasil dibuat!")
