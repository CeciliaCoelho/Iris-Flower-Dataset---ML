import sklearn.model_selection as model_selection
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


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

dectree = tree.DecisionTreeClassifier()
dectree = dectree.fit(X_train, y_train)
prediction = dectree.predict(X_test)


fn=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
cn=['Iris-setosa', 'Iris-virginica' , 'Iris-versicolor']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=200)
tree.plot_tree(dectree,
               feature_names = fn, 
               class_names=cn,
               filled = True)
plt.show()

accuracy=np.sum(y_test==prediction)/len(prediction) * 100 
print("Accuracy: ", accuracy )