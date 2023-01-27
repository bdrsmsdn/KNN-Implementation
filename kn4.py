import pandas as pd
import numpy as np

#Membaca data train dan data uji dari file excel
train_data = pd.read_excel("abalone.xlsx")
test_data = pd.read_excel("data_uji.xlsx")

# Memisahkan fitur (atribut) dan label dari data train dan data uji
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Menentukan jumlah tetangga terdekat (k) yang akan digunakan dalam model KNN
k = 59

# Function untuk menghitung jarak antara data uji dan data train
def compute_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2)**2))
    return distance

# Membuat list untuk menyimpan hasil prediksi
y_pred = []

# Loop untuk setiap data uji
for i in range(len(X_test)):
    # Membuat list untuk menyimpan jarak antara data uji dan data train
    distances = []
    # Loop untuk setiap data train
    for j in range(len(X_train)):
        # Menghitung jarak antara data uji dan data train
        distance = compute_distance(X_test[i], X_train[j])
        distances.append([distance, j])
    # Mengurutkan jarak dari yang terkecil
    distances = sorted(distances)
    # Membuat list untuk menyimpan kelas dari k tetangga terdekat
    classes = []
    # Loop untuk k tetangga terdekat
    for j in range(k):
        # Menambahkan kelas dari tetangga ke dalam list
        classes.append(y_train[distances[j][1]])
    # Menentukan kelas dari data uji berdasarkan kelas dari k tetangga terdekat
    y_pred.append(max(set(classes), key = classes.count))

# Menghitung akurasi
correct = 0
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        correct += 1
accuracy = correct / len(y_test)
print("Accuracy: ", accuracy)
# print((len(X_test)))