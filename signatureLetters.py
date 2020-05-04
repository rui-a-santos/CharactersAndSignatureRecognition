import sorting_character as sorting
import numpy as np
import random

#Creates an img numpy array
img = []

#Extracts signatures from Extract Letters
extract = sorting.Extract_Letters()

#Extracts the signatures from these files
s1 = extract.extractFile('signatures/r1.png')
s2 = extract.extractFile('signatures/r2.png')
s3 = extract.extractFile('signatures/m1.png')

#Appends the contents of the files to img numpy array
for i in range(len(s1)):
    img.append(np.array(s1)[i])

#Appends the contents of the files to img numpy array
for i in range(len(s2)):
    img.append(np.array(s2)[i])

#Appends the contents of the files to img numpy array
for i in range(len(s3)):
    img.append(np.array(s3)[i])

#Multiple variable declaraion - x is number of images, y is width, z is height
x, y, z = np.array(img).shape
#Reshape it to the length, then width*height
img = np.array(img).reshape(x, (y*z))

#Creates numpy array labels
labels = []

#Appends the 63 R signatures to labels
for i in range(63):
    labels.append("Rui")

#Appends the 80 M signatures to labels
for i in range(80):
    labels.append("Matt")

#Zips both the img and labels arrays together and shuffles them. (This keeps their label and img relation)
z = list(zip(img, labels))
random.shuffle(z)

#Unzips the 2 arrays
img, labels = zip(*z)

#Prints their shape
print(np.array(img).shape)
print(np.array(labels).shape)

#Gets 90% of the shuffled array for training and 10% for testing
x_train, x_test, y_train, y_test = img[:113], img[113:], labels[:113], labels[113:]

#Imports Libraries to calculate KNN
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_jobs=-1,weights='distance', n_neighbors=5)
knn_clf.fit(x_train, y_train)

#Prints the accuracy for testing
from sklearn.model_selection import cross_val_score
print("\nCross Validation Score:")
print(cross_val_score(knn_clf, x_test, y_test, cv=5, scoring="accuracy"))
arr = cross_val_score(knn_clf, x_test, y_test, cv=5, scoring="accuracy")

#Prints the average result for the cross validation testing
print("\nAverage result for the cross validation:", )
print(np.sum(arr)/5, "\n")
