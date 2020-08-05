import pandas as pd
import numpy as np
import sklearn.model_selection as model_selection
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

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

y = pd.DataFrame(y)
y = to_categorical(y , num_classes=3)

X_train, X_test , y_train , y_test = model_selection.train_test_split(X, y , train_size=0.80,test_size=0.20, random_state=101)


model = Sequential()
model.add(Dense(1000, input_dim=4, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=30, batch_size=10)

y_pred = model.predict(X_test)
y_label = np.argmax(y_test , axis=1)
predict_label = np.argmax(y_pred , axis=1)

accuracy = np.sum(y_label == predict_label) / len(y_pred) * 100 
print("Accuracy: ", accuracy )