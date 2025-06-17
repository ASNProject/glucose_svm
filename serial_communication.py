import serial
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import time

# =========================
# BACA DATA DARI ARDUINO
# =========================

# Ganti dengan port Arduino Anda (contoh: 'COM3' untuk Windows, '/dev/ttyUSB0' untuk Linux/Mac)
arduino_port = 'COM3'
baud_rate = 9600

# Inisialisasi koneksi serial
ser = serial.Serial(arduino_port, baud_rate, timeout=2)
time.sleep(2)  # Tunggu koneksi stabil

# Baca satu baris data
print("Menunggu data dari Arduino...")
line = ser.readline().decode('utf-8').strip()
ser.close()

# Parse data: contoh input -> "adc=300, glucose=100"
try:
    data_parts = dict(item.split("=") for item in line.split(","))
    new_adc = int(data_parts["adc"].strip())
    new_glucose = int(data_parts["glucose"].strip())
except Exception as e:
    print("Gagal parsing data dari Arduino:", e)
    exit()

print(f"Data dari Arduino -> ADC: {new_adc}, Glukosa: {new_glucose}")

# ===============
# Test SVR
# ===============
svr_model = joblib.load('svr_model.pkl')
scaler_factor = joblib.load('scaler_factor.pkl')

x_new = np.array([[new_adc, new_glucose]])
x_new_scaled = scaler_factor.transform(x_new)
factor_pred = svr_model.predict(x_new_scaled)[0]
result_pred = new_adc * factor_pred

print("\n=== Prediksi Faktor Pengali ===")
print(f"Faktor Prediksi : {factor_pred:.3f}")
print(f"Hasil Perkalian  : {result_pred:.2f}")

# ========================
# Test SVM Classification
# ========================
svm_diabetes = joblib.load('svm_diabetes_model.pkl')
scaler_diabetes = joblib.load('scaler_diabetes.pkl')

x_new_diab = np.array([[result_pred]])
x_diab_scaled = scaler_diabetes.transform(x_new_diab)

prediction = svm_diabetes.predict(x_diab_scaled)[0]
probability = svm_diabetes.predict_proba(x_diab_scaled)[0][1]

result = "Diabetes" if prediction == 1 else "Tidak Diabetes"
print("\n=== Prediksi Diabetes ===")
print(f"Hasil Prediksi : {result}")
print(f"Probabilitas   : {probability:.2%}")

# ========================
# Visualisasi
# ========================
df_diabetes = pd.read_excel('data.xlsx', sheet_name='Diabetes')
X_visual = df_diabetes[['glucose']].values
y_visual = df_diabetes['label'].values

X_visual_scaled = scaler_diabetes.transform(X_visual)

x_min, x_max = X_visual_scaled.min() - 1, X_visual_scaled.max() + 1
x_plot = np.linspace(x_min, x_max, 500).reshape(-1, 1)
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
