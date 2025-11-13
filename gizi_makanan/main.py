import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1️⃣ Baca dataset
data = pd.read_csv("menu.csv")

# 2️⃣ Tambahkan kolom baru 'Healthy' (1 = sehat, 0 = tidak sehat)
data['Healthy'] = data['Calories'].apply(lambda x: 1 if x < 400 else 0)

# 3️⃣ Pilih fitur (kolom gizi) untuk digunakan sebagai input model
features = [
    "Total Fat", "Saturated Fat", "Trans Fat", "Cholesterol",
    "Sodium", "Carbohydrates", "Sugars", "Protein"
]

X = data[features]
y = data["Healthy"]

# 4️⃣ Pisahkan data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Buat model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 6️⃣ Prediksi dan hitung akurasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Akurasi model: {accuracy:.2f}")

# 7️⃣ Coba prediksi manual
contoh = pd.DataFrame({
    "Total Fat": [10],
    "Saturated Fat": [3],
    "Trans Fat": [0],
    "Cholesterol": [30],
    "Sodium": [300],
    "Carbohydrates": [40],
    "Sugars": [12],
    "Protein": [8]
})

hasil = model.predict(contoh)
print("Prediksi contoh:", "Sehat" if hasil[0] == 1 else "Tidak Sehat")
