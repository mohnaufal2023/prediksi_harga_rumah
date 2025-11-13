import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Baca dataset
df = pd.read_csv('efek_sarapan.csv')

# Konversi kolom 'Rutin_Sarapan' ke bentuk numerik (Ya=1, Tidak=0)
df['Rutin_Sarapan'] = df['Rutin_Sarapan'].map({'Ya': 1, 'Tidak': 0})

# Fitur dan target
X = df[['Energi (1-10)', 'Konsentrasi (1-10)', 'Produktivitas (1-10)']]
y = df['Rutin_Sarapan']

# Bagi data jadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Buat model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Lakukan prediksi
y_pred = model.predict(X_test)

# Tampilkan hasil akurasi
print("Akurasi model: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nLaporan klasifikasi:")
print(classification_report(y_test, y_pred))

# Contoh prediksi data baru
data_baru = [[4, 5, 3]]  # Energi, Konsentrasi, Produktivitas
prediksi = model.predict(data_baru)
hasil = 'Rutin Sarapan' if prediksi[0] == 1 else 'Tidak Rutin Sarapan'

print("\nContoh prediksi untuk data (Energi=8, Konsentrasi=9, Produktivitas=8):")
print("➡️", hasil)
