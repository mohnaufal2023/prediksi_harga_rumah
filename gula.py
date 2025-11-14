import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# 1️⃣ Dataset simulasi 3 kelas
data = {
    'Sarapan': [1,0,1,0,1,0,1,0,1,0,
                0,0,0,1,1,1,0,0,1,1,
                0,1,0,1,0,1,0,1,0,1],
    'Berat_Badan': [65,70,60,75,68,72,62,74,66,73,
                    80,85,78,60,59,63,77,79,64,67,
                    55,63,72,66,75,60,78,62,70,65],
    'Aktivitas': [7,4,8,3,6,4,7,3,6,2,
                  2,1,3,8,7,7,2,1,6,7,
                  6,8,3,7,2,6,1,7,4,7],
    'Gula_Darah': ['Normal','Tinggi','Normal','Tinggi','Normal','Tinggi',
                   'Normal','Tinggi','Normal','Tinggi',
                   'Tinggi','Tinggi','Tinggi','Normal','Normal','Normal',
                   'Tinggi','Tinggi','Normal','Normal',
                   'Rendah','Normal','Tinggi','Normal','Tinggi','Normal','Tinggi','Normal','Tinggi','Normal']
}

df = pd.DataFrame(data)

# 2️⃣ Fitur & label
X = df[['Sarapan','Berat_Badan','Aktivitas']]
y = df['Gula_Darah']

# 3️⃣ Model Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# 4️⃣ Streamlit UI
st.title("Prediksi Gula Darah (Rendah / Normal / Tinggi)")

sarapan = st.radio("Apakah Anda sarapan hari ini?", (1,0), format_func=lambda x: "Ya" if x==1 else "Tidak")
berat = st.number_input("Masukkan berat badan Anda (kg):", min_value=30, max_value=150, value=65)
aktivitas = st.number_input("Berapa jam aktivitas/olahraga hari ini?", min_value=0, max_value=24, value=5)

if st.button("Prediksi"):
    user_input = pd.DataFrame({
        'Sarapan':[sarapan],
        'Berat_Badan':[berat],
        'Aktivitas':[aktivitas]
    })
    
    pred = model.predict(user_input)[0]
    st.subheader(f"Prediksi Gula Darah Anda: {pred}")
    
    # 5️⃣ Grafik batang sederhana
    probs = model.predict_proba(user_input)[0]
    categories = model.classes_
    prob_df = pd.DataFrame({'Kategori': categories, 'Probabilitas': probs})
    
    st.bar_chart(prob_df.set_index('Kategori'))
