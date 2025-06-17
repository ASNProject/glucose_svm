import tkinter as tk
from tkinter import messagebox
import serial
import threading
import time
import numpy as np
import joblib

# Load models dan scaler
svr_model = joblib.load('svr_model.pkl')
scaler_factor = joblib.load('scaler_factor.pkl')
svm_diabetes = joblib.load('svm_diabetes_model.pkl')
scaler_diabetes = joblib.load('scaler_diabetes.pkl')


# Fungsi klasifikasi
def classify_data(adc, glucose):
    try:
        x_new = np.array([[adc, glucose]])
        x_new_scaled = scaler_factor.transform(x_new)
        factor_pred = svr_model.predict(x_new_scaled)[0]
        result_pred = adc * factor_pred

        x_new_diab = np.array([[result_pred]])
        x_diab_scaled = scaler_diabetes.transform(x_new_diab)
        prediction = svm_diabetes.predict(x_diab_scaled)[0]
        probability = svm_diabetes.predict_proba(x_diab_scaled)[0][1]
        result = "Diabetes" if prediction == 1 else "Tidak Diabetes"
        return factor_pred, result_pred, result, probability
    except Exception as e:
        return None, None, f"Error: {e}", 0.0


# Realtime Reader
class SerialReader(threading.Thread):
    def __init__(self, port, update_callback):
        super().__init__(daemon=True)
        self.port = port
        self.update_callback = update_callback
        self.running = False
        self.ser = None

    def run(self):
        try:
            ser = serial.Serial(self.port, 9600, timeout=2)
            self.running = True
            time.sleep(2)  # Stabilize connection
            while self.running:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8').strip()
                    if "adc=" in line and "glucose=" in line:
                        parts = dict(item.split("=") for item in line.split(","))
                        adc = int(parts["adc"].strip())
                        glucose = int(parts["glucose"].strip())

                        factor, result_pred, result, prob = classify_data(adc, glucose)

                        if self.ser and self.ser.is_open and result_pred is not None:
                            result_str = f"{result_pred:.2f}\n"
                            self.ser.write(result_str.encode('utf-8'))

                        self.update_callback(adc, glucose)
                time.sleep(0.1)
        except Exception as e:
            self.update_callback(None, None, error=str(e))

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()

# UI App
class DiabetesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Realtime Diabetes Detection")
        self.serial_reader = None

        # UI Elements
        tk.Label(root, text="Port Serial:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.port_entry = tk.Entry(root)
        self.port_entry.insert(0, 'COM3')  # Default
        self.port_entry.grid(row=0, column=1, padx=5, pady=5)

        self.start_button = tk.Button(root, text="Start", command=self.start_serial)
        self.start_button.grid(row=0, column=2, padx=5, pady=5)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_serial, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=3, padx=5, pady=5)

        self.adc_label = tk.Label(root, text="ADC: -")
        self.adc_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        self.glucose_label = tk.Label(root, text="Glucose: -")
        self.glucose_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        self.multiplied_label = tk.Label(root, text="Hasil Perkalian: -")
        self.multiplied_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        self.result_label = tk.Label(root, text="Hasil: -", font=('Arial', 14, 'bold'))
        self.result_label.grid(row=3, column=0, columnspan=4, padx=5, pady=10)

    def update_display(self, adc, glucose, error=None):
        if error:
            self.result_label.config(text=f"Error: {error}", fg="red")
            return

        self.adc_label.config(text=f"ADC: {adc}")
        self.glucose_label.config(text=f"Glucose: {glucose}")
        factor, result_pred, result, prob = classify_data(adc, glucose)

        if result_pred is not None:
            self.multiplied_label.config(text=f"Hasil Perkalian: {result_pred:.2f}")
        else:
            self.multiplied_label.config(text=f"Hasil Perkalian: -")

        self.result_label.config(text=f"Hasil: {result} ({prob:.2%})",
                                 fg="green" if result == "Tidak Diabetes" else "red")

    def start_serial(self):
        port = self.port_entry.get()
        if not port:
            messagebox.showwarning("Warning", "Isi port serial terlebih dahulu!")
            return

        self.serial_reader = SerialReader(port, self.update_display)
        self.serial_reader.start()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

    def stop_serial(self):
        if self.serial_reader:
            self.serial_reader.stop()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.result_label.config(text="Hasil: -", fg="black")
        self.adc_label.config(text="ADC: -")
        self.glucose_label.config(text="Glucose: -")


# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesApp(root)
    root.mainloop()
