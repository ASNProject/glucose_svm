import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# ===============
# Test SVR
# ===============
new_adc = 300
new_glucose = 150

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
age = 42
glucose = 160
bmi = 29.0

# Load model
svm_diabetes = joblib.load('svm_diabetes_model.pkl')
scaler_diabetes = joblib.load('scaler_diabetes.pkl')

x_new_diab = np.array([[age, glucose, bmi]])
x_diab_scaled = scaler_diabetes.transform(x_new_diab)

# Prediction and Probability
prediction = svm_diabetes.predict(x_diab_scaled)[0]
probability = svm_diabetes.predict_proba(x_diab_scaled)[0][1]

result = "Diabetes" if prediction == 1 else "Tidak Diabetes"
print("=== Prediksi Diabetes ===")
print(f"Input: Umur={age}, Glukosa={glucose}, BMI={bmi}")
print(f"Hasil Prediksi : {result}")
print(f"Probabilitas   : {probability:.2%}")

# ========================
# Visualisasi SVM Diabetes (Glukosa & BMI)
# ========================
# Data dummy untuk visualisasi decision boundary
X_visual = np.array([
    [100, 22.0],
    [150, 28.5],
    [120, 26.1],
    [180, 31.2],
    [90, 20.5],
    [200, 35.3],
    [135, 24.0],
    [170, 30.1],
    [110, 23.3],
    [160, 29.7]
])
y_visual = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Gunakan scaler yang sama (fit sebelumnya)
X_visual_scaled = scaler_diabetes.transform(np.hstack((np.full((10,1), age), X_visual)))[:, 1:]  # hanya ambil Glukosa & BMI

# Buat meshgrid untuk decision boundary
h = .02
x_min, x_max = X_visual_scaled[:, 0].min() - 1, X_visual_scaled[:, 0].max() + 1
y_min, y_max = X_visual_scaled[:, 1].min() - 1, X_visual_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Ambil classifier dengan fitur Glukosa & BMI
clf_visual = SVC(kernel='rbf', probability=True)
clf_visual.fit(X_visual_scaled, y_visual)

Z = clf_visual.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot grafik
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
plt.scatter(X_visual_scaled[:, 0], X_visual_scaled[:, 1], c=y_visual, cmap=plt.cm.coolwarm, s=100, edgecolors='k')
plt.title("SVM Klasifikasi Diabetes (Glukosa & BMI)")
plt.xlabel("Glukosa (distandardisasi)")
plt.ylabel("BMI (distandardisasi)")
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_classification_diabetes.png')
plt.show()