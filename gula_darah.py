import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Dataset simulasi lebih besar dan variatif
data = {
    'Sarapan': [1,0,1,0,1,0,1,0,1,0, 0,0,0,1,1,1,0,0,1,1],
    'Berat_Badan': [65,70,60,75,68,72,62,74,66,73, 80,85,78,60,59,63,77,79,64,67],
    'Aktivitas': [7,4,8,3,6,4,7,3,6,2, 2,1,3,8,7,7,2,1,6,7],
    'Gula_Darah': ['Normal','Tinggi','Normal','Tinggi','Normal','Tinggi',
                   'Normal','Tinggi','Normal','Tinggi','Tinggi','Tinggi','Tinggi',
                   'Normal','Normal','Normal','Tinggi','Tinggi','Normal','Normal']
}

df = pd.DataFrame(data)

# 2️⃣ Fitur & label
X = df[['Sarapan','Berat_Badan','Aktivitas']]
y = df['Gula_Darah']

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Model Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Evaluasi model
y_pred = model.predict(X_test)
print("Akurasi model:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6️⃣ INTERAKTIF: Input user
print("\n===== Prediksi Gula Darah User =====")
sarapan = int(input("Apakah Anda sarapan hari ini? (1=Ya, 0=Tidak): "))
berat = float(input("Masukkan berat badan Anda (kg): "))
aktivitas = float(input("Berapa jam aktivitas/olahraga hari ini?: "))

user_input = pd.DataFrame({
    'Sarapan':[sarapan],
    'Berat_Badan':[berat],
    'Aktivitas':[aktivitas]
})

hasil = model.predict(user_input)
print("\nPrediksi gula darah Anda:", hasil[0])
