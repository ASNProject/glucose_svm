import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# =======================
# MULTIPLIER FACTOR
# =======================
adc = np.array([220.4, 344.9, 238.7, 310.2, 290.5, 265.3])
glucose = np.array([97, 174, 86, 145, 132, 122])
factor = glucose / adc
x_factor = np.column_stack((adc, glucose))
y_factor = factor

# Scaling
scaler_factor = StandardScaler()
x_factor_scaled = scaler_factor.fit_transform(x_factor)

# Model SVR
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
svr_model.fit(x_factor_scaled, y_factor)

# Save model & scaler
joblib.dump(svr_model, 'svr_model.pkl')
joblib.dump(scaler_factor, 'scaler_factor.pkl')

# Plot result training
y_pred = svr_model.predict(x_factor_scaled)
plt.plot(y_factor, label='Faktor Asli', marker='o')
plt.plot(y_pred, label="faktor Prediksi", marker='x')
plt.title('SVR - Prediksi Faktor Pengali')
plt.xlabel('Index')
plt.ylabel('Faktor Pengali')
plt.legend()
plt.grid()
plt.savefig('plot_svr_factor.png')
plt.show()

# ================================
# DIABETES MODEL
# ================================
# Dummy data
x_diabetes = np.array([
    [25, 100, 22.0],
    [45, 150, 28.5],
    [33, 120, 26.1],
    [50, 180, 31.2],
    [29, 90, 20.5],
    [60, 200, 35.3],
    [41, 135, 24.0],
    [48, 170, 30.1],
    [38, 110, 23.3],
    [55, 160, 29.7]
])
# Label: 0 = tidak diabetes, 1 = diabetes
y_diabetes = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# scaling
scaler_diabetes = StandardScaler()
x_diabetes_scaled = scaler_diabetes.fit_transform(x_diabetes)

# SVM Classifier
clf = SVC(kernel='rbf', probability=True)
clf.fit(x_diabetes_scaled, y_diabetes)

# Save model and scaler
joblib.dump(clf, 'svm_diabetes_model.pkl')
joblib.dump(scaler_diabetes, 'scaler_diabetes.pkl')

# Classification Report
X_train, X_test, y_train, y_test = train_test_split(x_diabetes_scaled, y_diabetes, test_size=0.2, random_state=42)
y_test_pred = clf.predict(X_test)
print("\n-- Classification Report ---")
print(classification_report(y_test, y_test_pred))