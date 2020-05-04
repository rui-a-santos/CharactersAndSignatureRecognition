from __future__ import division
import sorting_character as sorting
import numpy as np
import random
from matplotlib import pyplot as plt

#Gets the values in the folders
letters = sorting.Extract_Letters()

#Creates both arrays x is images, y is the labels
x = []
y = []

#Creates letter_data array
letter_data=[]

#Loads in the training data files
training_files = ['./ocr/training/training1.png', './ocr/training/training2.png','./ocr/training/training3.png','./ocr/training/training4.png','./ocr/training/training5.png','./ocr/training/training6.png']

#For every file in the training files get image data from the file
for files in training_files:
    letter_data = (letters.extractFile(files))
    print(np.array(letter_data).shape)

    #For every letter in the file it appends to x numpy array
    for letter in letter_data:
        x.append(letter)

#Declare what the alphabet looks like (no difference for capitals)
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9']

#Loops through all paragraphs in the 6  documents
for i in range (54):
    #Loops through all characters in a document's paragraph
    for j in range (61):
        #Appends the letter to the label
        y.append(alphabet[j])


#Zips x and y to not lose their relatioship
z = list(zip(x,y))

#Shuffles them
random.shuffle(z)

#Unzips it
x,y = zip(*z)

print("Total shape of the numpy array")
print(np.array(x).shape)

#Gets 90% of the shuffled array for training and 10% for testing
x_train, x_test, y_train, y_test = x[:2745], x[2745:], y[:2745], y[2745:]

#Reshapes the training and testing data to 28*28
x_train = np.array(x_train).reshape(2745,28*28)
x_test = np.array(x_test).reshape(549,28*28)

#Imports Libraries to calculate KNN
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_jobs=-1,weights='distance', n_neighbors=5)
knn_clf.fit(x_train, y_train)

#Prints the accuracy for testing
from sklearn.model_selection import cross_val_score
print("\nCross Validation Score of KNN:")
print(cross_val_score(knn_clf, x_test, y_test, cv=5, scoring="accuracy"))
arr = cross_val_score(knn_clf, x_test, y_test, cv=5, scoring="accuracy")

#Prints average result for cross validation
print("\nAverage result for the cross validation of KNN:", )
print(np.sum(arr)/5, "\n")

#Extracts letters from test file adobe.png
letters_test = letters.extractFile('./ocr/testing/adobe.png')
letters_test = np.array(letters_test).reshape(len(letters_test),28*28)
file = open('./ocr/testing/adobe_ground_truth.txt', encoding='utf8')

#Reads the file to test_data array
if file.mode == 'r':
    test_data = file.read()

#Creates a truth array
truth = []

#Takes out punctuation and spaces from extract letters
for i in test_data:
    if i.lower() in alphabet or i =='0'or i =='('or i ==')':
        truth.append(i)

#Returns an array of predictions from x test
y_knn_pred = knn_clf.predict(letters_test)
correct = 0

#Prints out the prediction and the true value
for i in range(len(y_knn_pred)  ):
    print("Predicted Letter: ", y_knn_pred[i], "   Actual Letter: ", truth[i])

#Checks if the character prediction is correct
for i in range (len(letters_test)):
    if np.array(y_knn_pred)[i].lower() == truth[i].lower():
        correct += 1

#Sets the correct percentage of the test file
percentage = (correct/len(truth))*100

#Prints out the correct guess percentage
print("\nThe correct guess percentage for the file using KNN is:")
print(percentage)
