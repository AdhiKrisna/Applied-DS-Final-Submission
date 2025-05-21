# 📊 Submission Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## 💼 Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan telah meluluskan banyak mahasiswa dengan prestasi gemilang. Akan tetapi, dalam perjalanannya, institusi ini menghadapi tantangan besar yaitu **tingginya angka dropout (DO)**. Fenomena ini berdampak negatif pada reputasi institusi dan efektivitas operasional pendidikan.

Sebagai calon data scientist, Anda ditugaskan untuk:

* Menganalisis data akademik dan demografis siswa
* Memprediksi kemungkinan dropout menggunakan teknik machine learning
* Membangun dashboard interaktif untuk membantu pihak kampus dalam pemantauan performa siswa

---

## ❓ Permasalahan Bisnis

1. Tingginya **angka mahasiswa yang dropout** tanpa bimbingan pencegahan sejak dini.
2. Minimnya pemahaman mengenai **faktor utama yang memengaruhi dropout**.
3. Belum adanya sistem visualisasi dan model prediktif untuk membantu pengambilan keputusan akademik secara proaktif.

---

## 📌 Cakupan Proyek

* Analisis eksploratif dan pembersihan data
* Transformasi dan encoding data kategorikal serta numerik
* Penanganan outliers dan ketidakseimbangan data
* Implementasi model machine learning (Random Forest, Logistic Regression, XGBoost)
* Deployment model dan visualisasi ke dalam dashboard interaktif dengan Streamlit
* Penyusunan insight dan rekomendasi berbasis data

---

## 🪡 Dataset dan Fitur

🔗 Link dataset: [Dicoding Student Performance Dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Dataset ini mencakup informasi mengenai data akademik, kehadiran, status keuangan, serta status akhir mahasiswa. Data terdiri dari:

### 🔢 Fitur Numerik:

* Application Order
* Previous Qualification Grade
* Admission Grade
* Age at Enrollment
* Curricular Units Semester 1 & 2 (credited, enrolled, approved, grades, etc)

### 📊 Fitur Ordinal:

* Marital Status

### 🛠️ Fitur Nominal:

* Application Mode
* Course
* Daytime/Evening Attendance
* Previous Qualification
* Nationality
* Gender
* Debtor
* Scholarship Holder
* Tuition Payment Up To Date
* International Student

### 🖳️ Kolom yang Dihapus:

* **GDP**, **Inflation Rate**, **Unemployment Rate**: kolom konstan, tidak informatif
* **Mothers\_occupation**, **Fathers\_occupation**, **Mothers\_qualification**, **Fathers\_qualification**: fitur yang tidak menunjukkan kontribusi signifikan terhadap prediksi dropout berdasarkan EDA dan feature importance

---

## ♻️ Preprocessing

* Mapping kategori ordinal dan nominal ke bentuk deskriptif
* Outlier handling dengan metode IQR + imputasi menggunakan KNNImputer
* Encoding:

  * Ordinal: `OrdinalEncoder`
  * Nominal: `OneHotEncoder`
  * Numerical: `StandardScaler`
* Split train-test dan balancing data dengan SMOTE

---

## 🎓 Modeling & Evaluation

Tiga model telah diuji:

1. **Random Forest Classifier**
2. Logistic Regression
3. XGBoost

### 📊 Best Model: Random Forest

Model ini menunjukkan akurasi dan interpretabilitas fitur terbaik di antara ketiganya. Dilengkapi dengan pipeline yang mengintegrasikan preprocessing dan training.

### 🔍 Feature Importance

* Fitur paling penting berasal dari performa mata kuliah semester 1 & 2
* Nilai masuk dan usia juga sangat berpengaruh
* Dukungan keuangan seperti beasiswa dan status pembayaran turut memberi dampak

---

## 📊 Business Dashboard

🔗 Live App (Streamlit): [https://student-dropout.streamlit.app](https://student-dropout.streamlit.app)

Tersedia 5 menu utama:

1. **Overview Data** – Tampilkan data mentah dan distribusi label
2. **Feature Importance** – Lihat fitur terpenting hasil model
3. **Perbandingan Fitur** – Bandingkan pengaruh fitur terhadap dropout
4. **Prediksi Dropout** – Form interaktif prediksi status siswa baru
5. **Insight & Rekomendasi** – Rangkuman hasil analisis & strategi

---

## ✅ Kesimpulan

Model machine learning berhasil mengidentifikasi siswa yang berisiko dropout berdasarkan data semester awal dan atribut demografis. Random Forest menunjukkan hasil terbaik dengan interpretasi yang kuat.

Dashboard Streamlit memudahkan institusi memantau performa siswa dan bertindak lebih dini terhadap potensi DO.

---

## 🚀 Rekomendasi Action Items

1. **Deteksi Dini dengan Kurikulum Semester Awal**
   Mahasiswa dengan performa semester awal rendah perlu dimasukkan ke program pendampingan belajar.

2. **Peningkatan Program Bimbingan untuk Mahasiswa Senior**
   Usia lebih tua saat masuk memiliki korelasi dengan dropout – butuh pendekatan adaptif.

3. **Optimalkan Sistem Beasiswa**
   Perluasan akses beasiswa dan kemudahan pembayaran terbukti menurunkan risiko dropout.

4. **Bangun Dashboard Monitoring**
   Gunakan dashboard ini sebagai alat monitoring performa akademik dan intervensi cepat.

5. **Validasi Ulang Jalur Masuk dan Program Studi**
   Beberapa jalur masuk atau prodi tertentu menunjukkan tren dropout yang lebih tinggi, perlu evaluasi.

---

## 💻 Repository Proyek

GitHub Repo: [https://github.com/AdhiKrisna/Applied-DS-Final-Submission.git](https://github.com/AdhiKrisna/Applied-DS-Final-Submission.git)

---

## ⚙️ Setup & Deploy

### 1. Clone Repository:

```bash
git clone https://github.com/AdhiKrisna/Applied-DS-Final-Submission.git
cd Applied-DS-Final-Submission
```

### 2. Setup Environment

```bash
python -m venv env
source env/bin/activate  # Windows: .\env\Scripts\activate
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

---

## 🔎 Prediksi Manual (Optional)

Jalankan `prediction.py` untuk tes prediksi lokal:

```bash
python prediction.py
```

Contoh output:

* ✅ Mahasiswa diprediksi **lulus**
* ⚠️ Mahasiswa diprediksi **dropout**

---

Terima kasih sudah membaca!

*"Early detection, better intervention. Save students, boost graduation rate!"*
