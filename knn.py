import sklearn.model_selection as model_selection
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X = []
y = []

data = pd.read_csv("C:/Users/Cecilia/Desktop/code/Iris_project/Iris.csv" , sep=",")

X = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

for row in data['Species']:

    if row == "Iris-setosa":
        y.append(0)
    elif row == "Iris-virginica":
        y.append(1)
    else:
        y.append(2)

X_train, X_test , y_train , y_test = model_selection.train_test_split(X, y , train_size=0.80,test_size=0.20, random_state=101)


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

score = np.mean(y_pred == y_test)
print("Model's score: " , score)