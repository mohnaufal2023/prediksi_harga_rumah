import pandas as pd

data = {
    'Nama': ['Andi', 'Budi', 'Citra', 'Dika', 'Eka', 'Fina', 'Gilang', 'Hani', 'Irfan', 'Joko'],
    'Umur': [20, 22, 21, 23, 20, 19, 24, 22, 21, 23],
    'Rutin_Sarapan': ['Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Ya', 'Tidak', 'Ya', 'Tidak', 'Tidak'],
    'Energi (1-10)': [8, 5, 9, 4, 7, 9, 5, 8, 6, 4],
    'Konsentrasi (1-10)': [9, 6, 8, 5, 8, 9, 6, 9, 6, 5],
    'Produktivitas (1-10)': [8, 5, 9, 5, 8, 9, 6, 8, 6, 5]
}

df = pd.DataFrame(data)
df.to_csv('efek_sarapan.csv', index=False)
print("✅ Dataset CSV berhasil dibuat!")
