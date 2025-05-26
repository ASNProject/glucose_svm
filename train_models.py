import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import pandas as pd

# =======================
# MULTIPLIER FACTOR
# =======================
df_factor = pd.read_excel('data.xlsx', sheet_name='Faktor')
adc = df_factor['adc'].values
glucose = df_factor['glucose'].values
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
df_diabetes = pd.read_excel('data.xlsx', sheet_name='Diabetes')
x_diabetes = df_diabetes[['glucose']].values
# Label: 0 = tidak diabetes, 1 = diabetes
y_diabetes = df_diabetes['label'].values

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