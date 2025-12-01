import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Unduh dan Muat Dataset
url = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/refs/heads/master/KDDTest-21.txt"

# Nama kolom sesuai dengan dataset NSL-KDD
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "target"
]

# Membaca dataset
try:
    data = pd.read_csv(url, header=None, names=column_names, on_bad_lines='skip')
    print("Dataset berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat dataset: {e}")
    exit()

# Tampilkan beberapa baris pertama dataset
print(data.head())

# 2. Preprocessing Dataset
# Pastikan semua kolom numerik yang mengandung string diubah
# Ganti nilai yang tidak sesuai (misalnya nilai yang tidak numerik) menjadi NaN atau angka tertentu
data['duration'] = pd.to_numeric(data['duration'], errors='coerce')
data['dst_host_srv_rerror_rate'] = pd.to_numeric(data['dst_host_srv_rerror_rate'], errors='coerce')

# Ganti NaN dengan nilai default (misalnya 0)
data['duration'].fillna(0, inplace=True)
data['dst_host_srv_rerror_rate'].fillna(0, inplace=True)

# Encode kolom kategorikal menjadi numerik
label_encoder = LabelEncoder()
categorical_cols = ['protocol_type', 'service', 'flag']

for col in categorical_cols:
    if data[col].dtype == 'object':  # Pastikan kolom memiliki tipe object
        data[col] = label_encoder.fit_transform(data[col])

# Konversi target menjadi biner (0 = normal, 1 = attack)
data['target'] = data['target'].apply(lambda x: 0 if x == "normal" else 1)

# Validasi ulang apakah semua kolom sudah numerik
non_numeric_cols = data.select_dtypes(include=['object']).columns
if not non_numeric_cols.empty:
    print(f"Kolom non-numerik terdeteksi: {non_numeric_cols}")
    print(data[non_numeric_cols].head())
    raise ValueError("Ada kolom non-numerik yang belum diubah ke bentuk numerik!")

# Pisahkan fitur dan target
X = data.drop(['target'], axis=1)
y = data['target']

# Standarisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 3. Membuat Model Neural Network
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Model berhasil dibuat.")

# 4. Melatih Model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

print("Model berhasil dilatih.")

# 5. Evaluasi Model
# Prediksi pada data pengujian
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Evaluasi akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy:.4f}")

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# 6. Visualisasi Hasil Pelatihan
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy During Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 7. Simpan Model
model.save("intrusion_detection_model.h5")
print("Model berhasil disimpan ke 'intrusion_detection_model.h5'.")
