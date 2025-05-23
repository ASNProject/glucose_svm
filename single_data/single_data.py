import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ================================
# Contoh Data Glukosa
# ================================
X = np.array([[100], [150], [120], [180], [90], [200], [130], [170], [110], [160]])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 0 = Tidak Diabetes, 1 = Diabetes

# ================================
# Scaling dan SVM
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = SVC(kernel='rbf', probability=True)
clf.fit(X_scaled, y)

# ================================
# Plot
# ================================
# Buat nilai glukosa yang lebih halus untuk prediksi
X_test = np.linspace(X_scaled.min()-1, X_scaled.max()+1, 500).reshape(-1, 1)
y_proba = clf.predict_proba(X_test)[:, 1]

plt.figure(figsize=(8, 4))
plt.plot(X_test, y_proba, color='red', label='Probabilitas Diabetes (kelas 1)')
plt.scatter(X_scaled[y == 0], [0]*len(y[y == 0]), color='blue', label='Tidak Diabetes')
plt.scatter(X_scaled[y == 1], [1]*len(y[y == 1]), color='red', label='Diabetes')
plt.title('SVM Klasifikasi Diabetes Berdasarkan Glukosa Saja')
plt.xlabel('Glukosa (scaled)')
plt.ylabel('Probabilitas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
