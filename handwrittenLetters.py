from __future__ import division
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
import tensorflow as tf
import sorting_character as sorting
import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as panda

#Reads in the balanced training data to x
x = panda.read_csv("./emnist-balanced-train.csv")

#Create digits array. Declare all numbers, caps letters and lower letters that are written differently to their caps version.
digits = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','e','f','g','h','n','q','r','t']

#Class gets every row, first column zero.
classes = x.values[:,0]
#Data gets every row, starting column after 1
x = x.values[:,1:]

#Creates y array
y=[]

#Flips the EMNIST data to be used
for i in range(len(x)):
    x[i] = np.transpose(x[i].reshape(28,28), axes=[1,0]).reshape(784)

#Appends the digits labels to array y
for i in classes:
    y.append(digits[i])

#Reshapes array with emnist to 28*28
x = np.array(x).reshape(len(x),28*28)
y = np.array(y).reshape(len(x))

#Gets 90% of the shuffled array for training and 10% for testing
x_train, x_test, y_train, y_test = x[:22500], x[22500:25000], y[:22500], y[22500:25000]

#Imports Libraries to calculate KNN
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_jobs=-1,weights='distance', n_neighbors=3)
knn_clf.fit(x_train, y_train)

#Prints the accuracy for testing
from sklearn.model_selection import cross_val_score
print("\nCross Validation Score for :")
print(cross_val_score(knn_clf, x_test, y_test, cv=5, scoring="accuracy"))
arr = cross_val_score(knn_clf, x_test, y_test, cv=5, scoring="accuracy")

#Prints average result for cross validation test
print("\nAverage result for the cross validation test for KNN:", )
print(np.sum(arr)/5)

#Puts it in onehot format
onehot = preprocessing.OneHotEncoder()

#Reshapes y based on length of y
y = np.array(y).reshape(len(y), 1)

#Puts it in onehot format
onehot_y = onehot.fit_transform(y)

#Reasigns y train and y test for CNN
y_train, y_test = onehot_y[:22500], onehot_y[22500:25000]

#Normalises data
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#Method adds to the model
def create_model():
    model = Sequential()
    #Adds convolutional layer with 64 nodes and 3x3 kernel. Activation function is relu
    model.add(Conv2D(64, kernel_size=3,activation="relu", input_shape=(28,28,1)))
    #Adds convolutional layer with 32 nodes and 3x3 kernel. Activation function is relu
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    #Defines the Dropout as 40%
    model.add(Dropout(0.4))
    #Flattens data
    model.add(Flatten())
    #Defines fully connected layer. 47 layers and softmax activation function.
    model.add(Dense(47, activation="softmax"))
    return model

#Cerates instance of model
model = create_model()

#Finding model features with adam being the optimizer and the loss function is categorical_crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Reshapes data
x_train = np.array(x_train).reshape(len(x_train),28,28,1)
x_test = np.array(x_test).reshape(len(x_test),28,28,1)

#Run model witht he data
model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=10, batch_size=64)
