# Intrusion Detection System

Ini adalah proyek untuk membangun sistem deteksi intrusi menggunakan dataset NSL-KDD. Model yang digunakan adalah neural network untuk mengklasifikasikan data sebagai "normal" atau "attack" berdasarkan fitur yang diberikan.

## Struktur Proyek

- **Data Loss Prevention.py**: Script utama untuk memuat dataset, melakukan preprocessing, membangun model neural network, melatih model, mengevaluasi model, dan menyimpan model.
- **intrusion_detection_model.h5**: Model yang sudah dilatih dan disimpan.

## Prerequisites

Pastikan untuk menginstal dependensi berikut:
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib

Install dependensi menggunakan pip:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
Cara Penggunaan
Jalankan script Data Loss Prevention.py untuk memulai proses deteksi intrusi.

Model akan dilatih menggunakan dataset NSL-KDD dan hasil pelatihan akan divisualisasikan.

Setelah pelatihan selesai, model akan disimpan dalam file intrusion_detection_model.h5.

Dataset
Dataset yang digunakan dalam proyek ini adalah NSL-KDD yang dapat diunduh dari:

URL: https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/refs/heads/master/KDDTest-21.txt

Evaluasi
Setelah model dilatih, performa model dievaluasi menggunakan akurasi, classification report, dan confusion matrix.

Visualisasi
Selama pelatihan, grafik mengenai loss dan akurasi akan ditampilkan untuk menunjukkan perkembangan model.

Lisensi
Proyek ini dilisensikan di bawah MIT License.
