import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Dropout Dashboard", layout="wide")

# Load data dan model
@st.cache_data
def load_data():
    return pd.read_csv("data/explored_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("model/rf_model.pkl")

@st.cache_resource
def load_preprocessor():
    return joblib.load("model/preprocessor.pkl")

df = load_data()
model = load_model()
preprocessor = load_preprocessor()

# Pengelompokan fitur
numeric_features = [
    'Application_order', 'Previous_qualification_grade', 'Admission_grade',
    'Age_at_enrollment', 'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations'
]

nominal_cols = [
    'Application_mode', 'Course', 'Daytime_evening_attendance',
    'Previous_qualification', 'Nacionality',
    'Mothers_occupation', 'Fathers_occupation',
    'Displaced', 'Educational_special_needs', 'Debtor',
    'Tuition_fees_up_to_date', 'Gender',
    'Scholarship_holder', 'International'
]

ordinal_cols = ['Marital_status', 'Mothers_qualification', 'Fathers_qualification']

# Fungsi visualisasi fitur kategorikal
def plot_categorical_feature(df, col):
    if len(df[col].unique()) > 10:
        plt.figure(figsize=(12, 6))
        plt.xticks(rotation=45, ha='right')
    else:
        plt.figure(figsize=(8, 4))
        plt.xticks(rotation=45)
    sns.countplot(x=col, hue='Status', data=df)
    plt.title(f'Sebaran Status Berdasarkan {col}')
    plt.xticks(rotation=60)
    plt.tight_layout()
    st.pyplot(plt.gcf())

# Sidebar Navigation
menu = st.sidebar.radio("Menu", [
    "Data Overview",
    "Feature Importance",
    "Perbandingan Fitur",
    "Prediksi Dropout",
    "Insights & Recommendations"
])

# === MENU 1 ===
if menu == "Data Overview":
    st.title("📊 Gambaran Data Mahasiswa")
    st.dataframe(df)
    st.write("Jumlah data:", df.shape[0])
    st.write("Jumlah fitur:", df.shape[1])

    fig, ax = plt.subplots()
    sns.countplot(x='Status', data=df, ax=ax)
    ax.set_title('Distribusi Status Mahasiswa')
    st.pyplot(fig)

# === MENU 2 ===
elif menu == "Feature Importance":
    st.title("🔍 Feature Importance")
    if hasattr(model, 'feature_importances_'):
        try:
            feature_names = preprocessor.get_feature_names_out()
            importances = model.feature_importances_
            feature_names = feature_names[:len(importances)]

            imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            st.dataframe(imp_df)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=imp_df.head(10), x='Importance', y='Feature', ax=ax)
            ax.set_title('Top 10 Important Features')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal memuat feature importance: {e}")
    else:
        st.warning("Model tidak memiliki atribut feature_importances_.")

# === MENU 3 ===
elif menu == "Perbandingan Fitur":
    st.title("📊 Komparasi Fitur terhadap Status Mahasiswa")
    selected_feature = st.selectbox("Pilih fitur:", df.columns.drop(['Status', 'Inflation_rate', 'GDP', 'Unemployment_rate', 'Mothers_occupation', 'Fathers_occupation', 'Mothers_qualification', 'Fathers_qualification']))
    if selected_feature in nominal_cols or selected_feature in ordinal_cols:
        plot_categorical_feature(df, selected_feature)
    elif selected_feature in numeric_features:
        fig, ax = plt.subplots()
        sns.boxplot(x='Status', y=selected_feature, data=df, ax=ax)
        ax.set_title(f"{selected_feature} vs Status")
        st.pyplot(fig)
    else:
        st.warning("Tipe fitur tidak dikenali untuk visualisasi.")

# === MENU 4 ===
elif menu == "Prediksi Dropout":
    st.title("🎯 Prediksi Dropout Mahasiswa")
    user_input = {}

    excluded_columns = [
        'Status', 'Inflation_rate', 'GDP', 'Unemployment_rate',
        'Mothers_occupation', 'Fathers_occupation',
        'Mothers_qualification', 'Fathers_qualification'
    ]

    prediction_features = [col for col in df.columns if col not in excluded_columns]

    for col in prediction_features:
        if df[col].dtype == 'object':
            user_input[col] = st.selectbox(col, options=sorted(df[col].dropna().unique()))
        else:
            user_input[col] = st.number_input(col, value=float(df[col].mean()))

    if st.button("Prediksi Dropout"):
        input_df = pd.DataFrame([user_input])
        input_df = input_df[prediction_features]

        try:
            transformed_input = preprocessor.transform(input_df)
            prediction = model.predict(transformed_input)[0]
            if prediction == 0:
                st.warning("⚠️ Mahasiswa berisiko dropout!")
            elif prediction == 1:
                st.success("✅ Mahasiswa sedang terdaftar.")
            else:
                st.success("🎓 Mahasiswa diprediksi lulus (graduated)")
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")

# === MENU 5 ===
elif menu == "Insights & Recommendations":
    st.title("💡 Insight & Rekomendasi")
    st.markdown("""
    Berdasarkan hasil modeling dan analisis, berikut beberapa insight utama:

    1. **📚 Kurikulum Semester Awal** — Mahasiswa dengan performa rendah (unit sedikit/tidak lulus) cenderung berpotensi dropout.
    2. **🎯 Umur & Nilai Masuk** — Usia lebih tua dan nilai rendah saat diterima → peningkatan risiko dropout.
    3. **💸 Beasiswa & Biaya Kuliah** — Mahasiswa dengan beasiswa dan pembayaran lancar → lebih besar peluang lulus.

    """)

    st.markdown("## 🚀 Rekomendasi Action Items")
    st.markdown("""
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
    """)
