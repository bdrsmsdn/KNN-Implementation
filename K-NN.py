import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
dataset = pd.read_excel('abalone.xlsx')
data = dataset[['Length', 'Diameter', 'Height', 'Whole weight', 'Shell weight', 'Rings', 'ClassSex']]
data = data.dropna()
train_data = data[['Length', 'Diameter', 'Height', 'Whole weight', 'Shell weight', 'Rings']]
train_label = data[['ClassSex']]
kNN=KNeighborsClassifier(n_neighbors=3, weights='distance')
test_data = [[5.6, 4.6, 2.4, 4.9, 2.9, 15]]
kNN.fit(train_data, np.ravel(train_label))
class_result=kNN.predict(test_data)
print('Hasil Klasifikasi: ', class_result.item())