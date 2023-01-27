import pandas as pd
import numpy as np

# Fungsi untuk mencari jarak euclidean
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Fungsi untuk menentukan kelas dari data uji
def predict(X_train, y_train, x_test, k):
    distances = []
    for i in range(X_train.shape[0]):
        distance = euclidean_distance(x_test, X_train[i])
        distances.append([distance, i])
    distances = sorted(distances)
    targets = []
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])
    return max(targets, key=targets.count)

# Baca data latih dari file excel
df = pd.read_excel('abalone.xlsx')
X_train = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shell weight', 'Rings']].to_numpy()
y_train = df['ClassSex'].to_numpy()

# Baca data uji dari file excel
df = pd.read_excel('data_uji.xlsx')
X_test = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shell weight', 'Rings']].to_numpy()
y_test = df['ClassSex'].to_numpy()

# Tentukan kelas dari data uji dengan jumlah tetangga terdekat sebanyak 3
k = 59
predictions = []
for x_test in X_test:
    predictions.append(predict(X_train, y_train, x_test, k))

# Hitung akurasi
accuracy = np.mean(predictions == y_test)
print("Akurasi model: ", accuracy)