import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd

# MASUKKAN DATA BARU
new_adc = 300
new_glucose = 150

# ===============
# Test SVR
# ===============
# Load model
svr_model = joblib.load('svr_model.pkl')
scaler_factor = joblib.load('scaler_factor.pkl')

x_new = np.array([[new_adc, new_glucose]])
x_new_scaled = scaler_factor.transform(x_new)
factor_pred = svr_model.predict(x_new_scaled)[0]
result_pred = new_adc * factor_pred

print("=== Prediksi Faktor Pengali ===")
print(f"ADC: {new_adc}, Darah: {new_glucose}")
print(f"Faktor Prediksi : {factor_pred:.3f}")
print(f"Hasil Perkalian  : {result_pred:.2f}\n")


# ========================
# Test SVM Classification
# ========================

# Load model
svm_diabetes = joblib.load('svm_diabetes_model.pkl')
scaler_diabetes = joblib.load('scaler_diabetes.pkl')

x_new_diab = np.array([[result_pred]])
x_diab_scaled = scaler_diabetes.transform(x_new_diab)

# Prediction and Probability
prediction = svm_diabetes.predict(x_diab_scaled)[0]
probability = svm_diabetes.predict_proba(x_diab_scaled)[0][1]

result = "Diabetes" if prediction == 1 else "Tidak Diabetes"
print("=== Prediksi Diabetes ===")
print(f"Input: Glukosa = {result_pred:.2f}")
print(f"Hasil Prediksi : {result}")
print(f"Probabilitas   : {probability:.2%}")

# ========================
# Visualisasi SVM Diabetes (Glukosa)
# ========================
# Data asli
df_diabetes = pd.read_excel('data.xlsx', sheet_name='Diabetes')
X_visual = df_diabetes[['glucose']].values
y_visual = df_diabetes['label'].values

# Load scaler dan model kamu
scaler_diabetes = joblib.load('scaler_diabetes.pkl')
svm_diabetes = joblib.load('svm_diabetes_model.pkl')

# Scaling fitur glucose saja
X_visual_scaled = scaler_diabetes.transform(X_visual)

# Buat range untuk plot decision function
x_min, x_max = X_visual_scaled.min() - 1, X_visual_scaled.max() + 1
x_plot = np.linspace(x_min, x_max, 500).reshape(-1, 1)

# Prediksi probabilitas menggunakan model
probs = svm_diabetes.predict_proba(x_plot)[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(X_visual_scaled, y_visual, c=y_visual, cmap=plt.cm.coolwarm, s=100, edgecolors='k')
plt.plot(x_plot, probs, 'b-', label='Probabilitas Diabetes')
plt.axhline(0.5, color='gray', linestyle='--', label='Threshold 0.5')
plt.title('SVM Diabetes Classification (Glucose only)')
plt.xlabel('Glukosa (standarisasi)')
plt.ylabel('Probabilitas Diabetes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_classification_diabetes.png')
plt.show()