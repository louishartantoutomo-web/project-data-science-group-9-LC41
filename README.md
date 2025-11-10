# Dashboard Clustering Kondisi Mesin - Predictive Maintenance

Dashboard interaktif berbasis Streamlit untuk visualisasi dan prediksi clustering kondisi mesin menggunakan K-Means.

## 📋 Fitur Utama

1. **Overview Dashboard** - Ringkasan dataset dan distribusi cluster
2. **Visualisasi 3D Interaktif** - Grafik 3D yang dapat diputar dan dizoom
3. **Prediksi Cluster Real-time** - Input data baru dan dapatkan prediksi cluster
4. **Analisis Detail Cluster** - Statistik dan karakteristik setiap cluster

## 🚀 Cara Instalasi & Menjalankan

### 1. Persiapan Environment

Pastikan Anda sudah menginstal Python 3.8 atau lebih baru. Kemudian install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Simpan Model

Sebelum menjalankan dashboard, Anda perlu menyimpan model K-Means dan scaler terlebih dahulu.

**Opsi A: Menggunakan script save_model.py**

Pastikan file `predictive_maintenance.csv` ada di folder yang sama, lalu jalankan:

```bash
python save_model.py
```

**Opsi B: Dari notebook Jupyter**

Tambahkan kode berikut di akhir notebook Anda setelah melatih model:

```python
import joblib

# Simpan model dan scaler
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Simpan data clustering
df_clustered = df.copy()
df_clustered['Cluster'] = clusters
df_clustered.to_csv('clustered_data.csv', index=False)

print("✅ Model berhasil disimpan!")
```

### 3. Jalankan Dashboard

Setelah model tersimpan, jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```

Dashboard akan terbuka otomatis di browser Anda (biasanya di `http://localhost:8501`)

## 📁 Struktur File

```
.
├── app.py                          # Aplikasi Streamlit utama
├── save_model.py                   # Script untuk menyimpan model
├── requirements.txt                # Dependencies Python
├── predictive_maintenance.csv      # Dataset asli (perlu Anda sediakan)
├── kmeans_model.pkl               # Model K-Means (otomatis dibuat)
├── scaler.pkl                     # StandardScaler (otomatis dibuat)
└── clustered_data.csv             # Data dengan label cluster (otomatis dibuat)
```

## 🎯 Cara Menggunakan Dashboard

### 1. Menu Overview
- Lihat ringkasan dataset
- Distribusi data per cluster
- Statistik deskriptif variabel

### 2. Menu Visualisasi 3D
- Grafik 3D interaktif menampilkan semua data points
- **Cara interaksi:**
  - Klik dan drag untuk memutar grafik
  - Scroll untuk zoom in/out
  - Hover pada titik untuk melihat detail
- Atur ukuran titik dan transparansi
- Tampilkan/sembunyikan centroid cluster

### 3. Menu Prediksi Cluster
- Input 3 variabel:
  - Torque (Nm)
  - Process Temperature (K)
  - Tool Wear (min)
- Klik tombol "Prediksi Cluster"
- Sistem akan menampilkan:
  - Cluster terprediksi
  - Tingkat kepercayaan prediksi
  - Karakteristik cluster
  - Posisi data input dalam visualisasi 3D

### 4. Menu Analisis Cluster
- Pilih cluster yang ingin dianalisis
- Lihat statistik detail cluster
- Distribusi variabel dalam cluster
- Perbandingan dengan rata-rata keseluruhan
- Distribusi jenis failure

## 📊 Variabel Input

Dashboard menggunakan 3 variabel untuk clustering:

1. **Torque [Nm]** - Torsi/momen gaya mesin
2. **Process Temperature [K]** - Suhu proses mesin dalam Kelvin
3. **Tool Wear [min]** - Waktu penggunaan alat

## 🔧 Kustomisasi

### Mengubah Jumlah Cluster

Jika Anda ingin mengubah jumlah cluster, edit file `save_model.py`:

```python
optimal_k = 8  # Ubah angka ini sesuai kebutuhan
```

Kemudian jalankan ulang:
```bash
python save_model.py
streamlit run app.py
```

### Mengubah Variabel Clustering

Edit array `features` di `save_model.py` dan `app.py`:

```python
features = ['Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
```

## ⚠️ Troubleshooting

### Error: Model tidak ditemukan
**Solusi:** Jalankan `python save_model.py` terlebih dahulu

### Error: Data tidak ditemukan
**Solusi:** Pastikan file `predictive_maintenance.csv` ada di folder yang sama

### Dashboard tidak terbuka
**Solusi:** 
1. Cek apakah port 8501 sudah digunakan
2. Coba buka manual di browser: `http://localhost:8501`
3. Gunakan port lain: `streamlit run app.py --server.port 8502`

### Visualisasi 3D lambat
**Solusi:** Kurangi ukuran titik atau transparansi di menu Visualisasi 3D

## 📝 Catatan Penting

- Pastikan dataset sudah bersih sebelum digunakan
- Model perlu dilatih ulang jika ada perubahan data signifikan
- Dashboard ini dirancang untuk 8 cluster (sesuai hasil optimal K-Means)
- Semua prediksi menggunakan data yang sudah dinormalisasi

## 👥 Developer

Dashboard ini dikembangkan sebagai bagian dari project Machine Learning untuk Predictive Maintenance:

- **Kelompok 9 (LC41)**
- LOUIS HARTANTO UTOMO - 2702285744
- RAYMOND CHRISTOPHER SOFIAN - 2702320482
- GELFAND HANLI LIM - 2702322071
- KARLINA GUNAWAN - 2702252973

## 📞 Support

Jika ada pertanyaan atau masalah, silakan hubungi tim developer.

---

**Selamat menggunakan Dashboard Clustering Kondisi Mesin! 🎉**
