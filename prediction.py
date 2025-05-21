import pandas as pd
import joblib

# Load model dan preprocessor
model = joblib.load("model/rf_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")

# Simulasi input data baru
sample_input = {
    'Application_mode': "1st phase - general contingent",
    'Course': "Informatics Engineering",
    'Daytime_evening_attendance': "Daytime",
    'Previous_qualification': "Secondary Edu",
    'Nacionality': "Portuguese",
    'Displaced': "No",
    'Educational_special_needs': "No",
    'Debtor': "No",
    'Tuition_fees_up_to_date': "Yes",
    'Gender': "Male",
    'Scholarship_holder': "No",
    'International': "No",
    'Marital_status': "Single",
    'Application_order': 1,
    'Previous_qualification_grade': 120.0,
    'Admission_grade': 140.0,
    'Age_at_enrollment': 18,
    'Curricular_units_1st_sem_credited': 0,
    'Curricular_units_1st_sem_enrolled': 6,
    'Curricular_units_1st_sem_evaluations': 6,
    'Curricular_units_1st_sem_approved': 5,
    'Curricular_units_1st_sem_grade': 14.0,
    'Curricular_units_1st_sem_without_evaluations': 0,
    'Curricular_units_2nd_sem_credited': 0,
    'Curricular_units_2nd_sem_enrolled': 6,
    'Curricular_units_2nd_sem_evaluations': 6,
    'Curricular_units_2nd_sem_approved': 6,
    'Curricular_units_2nd_sem_grade': 15.0,
    'Curricular_units_2nd_sem_without_evaluations': 0
}

# Buat DataFrame dari input
input_df = pd.DataFrame([sample_input])

# Transformasi fitur dengan preprocessor
X_transformed = preprocessor.transform(input_df)

# Prediksi
prediction = model.predict(X_transformed)[0]
label = "Dropout" if prediction == 0 else "Enrolled" if prediction == 1 else "Graduate"

print("\n=== Prediksi Dropout Mahasiswa ===")
print("Input Mahasiswa:")
print(input_df.T)
print("\nHasil Prediksi:")
if prediction == 0:
    print("Mahasiswa diprediksi: Dropout")
elif prediction == 1:
    print("Mahasiswa diprediksi: Enrolled")
else:
    print("Mahasiswa diprediksi: Graduate")
