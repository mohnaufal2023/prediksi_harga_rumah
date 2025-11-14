import pandas as pd
import numpy as np

# jumlah data
N = 2000

np.random.seed(42)

data = {
    "Sarapan": np.random.choice([0, 1], size=N),
    "Berat_Badan": np.random.randint(45, 100, size=N),
    "Aktivitas": np.random.randint(0, 10, size=N)
}

df = pd.DataFrame(data)

# aturan realistis untuk label gula darah
labels = []
for i in range(N):
    s = df.loc[i, "Sarapan"]
    b = df.loc[i, "Berat_Badan"]
    a = df.loc[i, "Aktivitas"]

    if s == 0 and a <= 2:
        labels.append("Tinggi")
    elif s == 0 and a >= 7:
        labels.append("Rendah")
    elif b >= 90:
        labels.append("Tinggi")
    elif b <= 50:
        labels.append("Rendah")
    else:
        labels.append("Normal")

df["Gula_Darah"] = labels

df.to_csv("dataset_gula.csv", index=False)
print("Dataset 2000 baris berhasil dibuat → dataset_gula.csv")
