import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv("gula_dataset_1000.csv")

X = df[['Sarapan','Berat_Badan','Aktivitas','Usia','Tidur','Gula_Harian','Riwayat_Keluarga']]
y = df['Gula_Darah']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"Akurasi Model: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, pred))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel telah disimpan ke model.pkl")
