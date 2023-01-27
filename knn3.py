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

# menghitung jarak euclidean
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

y_pred = []
# looping untuk setiap data uji
for i in range(len(X_test)):
    distances = []
    # looping untuk setiap data train
    for j in range(len(X_train)):
        # menghitung jarak antara data uji dan data train
        distance = euclidean_distance(X_test[i], X_train[j])
        distances.append([distance, j])
    # mengurutkan jarak dari yang terdekat
    distances = sorted(distances)
    target = []
    # menentukan tetangga terdekat
    for l in range(k):
        index = distances[l][1]
        target.append(y_train[index])
    # menentukan label dari data uji dengan menghitung mayoritas label dari tetangga terdekat
    y_pred.append(max(set(target), key = target.count))

correct = 0
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        correct += 1
accuracy = correct / len(y_test)
print("Accuracy:", accuracy)

# # Menghitung jumlah true positive, true negative, false positive, dan false negative
# tp = 0
# tn = 0
# fp = 0
# fn = 0
# for i in range(len(y_test)):
#     if y_test[i] == y_pred[i] == 1:
#         tp += 1
#     elif y_test[i] == y_pred[i] == 0:
#         tn += 1
#     elif y_test[i] == 0 and y_pred[i] == 1:
#         fp += 1
#     elif y_test[i] == 1 and y_pred[i] == 0:
#         fn += 1

# # Menghitung precision, recall, dan f1_score
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# f1_score = 2 * (precision * recall) / (precision + recall)

# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1_score)
# Menghitung jumlah true positive, true negative, false positive, dan false negative
# tp = 0
# tn = 0
# fp = 0
# fn = 0
# for i in range(len(y_test)):
#     if y_test[i] == y_pred[i] == 1:
#         tp += 1
#     elif y_test[i] == y_pred[i] == 0:
#         tn += 1
#     elif y_test[i] == 0 and y_pred[i] == 1:
#         fp += 1
#     elif y_test[i] == 1 and y_pred[i] == 0:
#         fn += 1

# # Menghitung precision, recall, dan f1_score
# if tp+fp == 0:
#     precision = 0
# else:
#     precision = tp / (tp + fp)
# if tp+fn == 0:
#     recall = 0
# else:
#     recall = tp / (tp + fn)
# if precision+recall == 0:
#     f1_score = 0
# else:
#     f1_score = 2 * (precision * recall) / (precision + recall)

# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1_score)
