from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Data dasar (sama kayak dataset sebelumnya)
data = {
    'Energi (1-10)': [8, 5, 9, 4, 7, 9, 5, 8, 6, 4],
    'Konsentrasi (1-10)': [9, 6, 8, 5, 8, 9, 6, 9, 6, 5],
    'Produktivitas (1-10)': [8, 5, 9, 5, 8, 9, 6, 8, 6, 5],
    'Rutin_Sarapan': [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

# Model ML sederhana
X = df[['Energi (1-10)', 'Konsentrasi (1-10)', 'Produktivitas (1-10)']]
y = df['Rutin_Sarapan']
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    energi = int(request.form['energi'])
    konsentrasi = int(request.form['konsentrasi'])
    produktivitas = int(request.form['produktivitas'])
    
    hasil = model.predict([[energi, konsentrasi, produktivitas]])[0]
    hasil_teks = 'Rutin Sarapan 🍞' if hasil == 1 else 'Tidak Rutin Sarapan ☕'
    
    return render_template('index.html', hasil=hasil_teks)

if __name__ == '__main__':
    app.run(debug=True)
