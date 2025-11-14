import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# load dataset
df = pd.read_csv("dataset_gula.csv")

X = df[['Sarapan','Berat_Badan','Aktivitas']]
y = df['Gula_Darah']

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# prediksi
pred = model.predict(X_test)

# akurasi
print("Akurasi:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# save model
joblib.dump(model, "model_gula.pkl")
print("\nModel tersimpan sebagai model_gula.pkl")
