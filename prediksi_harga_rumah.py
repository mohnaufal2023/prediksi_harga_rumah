# --- Prediksi Harga Rumah Sederhana ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Dataset sederhana
data = {
    'Luas': [50, 60, 70, 80, 90, 100, 110, 120],
    'Kamar': [2, 3, 3, 4, 4, 5, 5, 6],
    'Harga': [500, 600, 650, 800, 850, 950, 1000, 1100]  # dalam juta
}

df = pd.DataFrame(data)
print("Data Rumah:\n", df, "\n")

# 2. Pisahkan fitur (X) dan target (y)
X = df[['Luas', 'Kamar']]
y = df['Harga']

# 3. Split data jadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Buat model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Tampilkan hasil training
print("Koefisien (pengaruh tiap fitur):", model.coef_)
print("Intercept (nilai dasar):", model.intercept_, "\n")

# 6. Uji model dengan data test
prediksi = model.predict(X_test)
hasil = pd.DataFrame({'Asli': y_test, 'Prediksi': prediksi})
print("Hasil Prediksi:\n", hasil, "\n")

# 7. Visualisasi hasil
plt.scatter(df['Luas'], df['Harga'], color='blue', label='Data Asli')
plt.xlabel('Luas Rumah (m²)')
plt.ylabel('Harga (juta)')
plt.title('Hubungan Luas dan Harga Rumah')
plt.legend()
plt.show()

# 8. Prediksi rumah baru
luas_baru = 95
kamar_baru = 4
harga_prediksi = model.predict([[luas_baru, kamar_baru]])
print(f"Prediksi harga rumah dengan luas {luas_baru} m² dan {kamar_baru} kamar adalah {harga_prediksi[0]:.2f} juta")
