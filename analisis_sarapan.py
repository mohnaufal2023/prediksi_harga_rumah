import pandas as pd
import matplotlib.pyplot as plt

# Membaca dataset CSV
df = pd.read_csv('efek_sarapan.csv')

# Menampilkan data awal
print("Data awal:")
print(df)

# Menghitung rata-rata berdasarkan kebiasaan sarapan
rata2 = df.groupby('Rutin_Sarapan')[['Energi (1-10)', 'Konsentrasi (1-10)', 'Produktivitas (1-10)']].mean()
print("\nRata-rata berdasarkan kebiasaan sarapan:")
print(rata2)

# Visualisasi perbandingan
rata2.plot(kind='bar', figsize=(8,5), color=['skyblue', 'salmon'])
plt.title('Perbandingan Rata-rata antara yang Sarapan dan Tidak Sarapan')
plt.ylabel('Skor (1-10)')
plt.xlabel('Kebiasaan Sarapan')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
