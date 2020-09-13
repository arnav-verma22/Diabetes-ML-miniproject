# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:27:48 2020

@author: Arnav Verma
"""
import numpy as np
import pandas as pd

#importing Dataset
df = pd.read_csv("datasets_228_482_diabetes.csv")
y = df['Outcome']
x = df.drop('Outcome', axis=1)

#removing null values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 0, strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(x)
x = pd.DataFrame(imputer.transform(x), columns=x.columns)

#splitting into train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#applying KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

#prdicting data using KNN
y_pred = classifier.predict(x_test)

#Checking score of KNN
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred)
print("Accuracy of KNN Algorithm: ",classifier.score(x_test, y_test)*100,"%")

#aplying decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

#prdicting data using Decision Tree
y_pred = classifier.predict(x_test)

#Checking score of Decision Tree Algorithm
cm_DT = confusion_matrix(y_test, y_pred)
print("Accuracy of Decision Tree Algorithm: ",classifier.score(x_test, y_test)*100,"%")

#Predicting on Custom Dataset
a = {'Pregnancies':[0], 'Glucose':[120], 'BloodPressure':[70], 'SkinThickness':[27], 'Insulin':[135], 'BMI':[26], 'DiabetesPedigreeFunction':[0.4], 'Age':[20]}
test = pd.DataFrame.from_dict(a)
predict = classifier.predict(test)

#plotting Histogram
df.hist(bins=30, figsize=(15, 10))
#ax = x.plot.hist(bins=12, alpha=0.5)

#plotting scatter plot
ax1 = df.plot.scatter(x='Glucose', y='Insulin', c='DarkBlue')
ax2 = df.plot.scatter(x='BMI', y='Glucose', c='DarkBlue')
ax3 = df.plot.scatter(x='BloodPressure', y='Insulin', c='DarkBlue')
ax4 = df.plot.scatter(x='BMI', y='BloodPressure', c='DarkBlue')
ax5 = df.plot.scatter(x='SkinThickness', y='Insulin', c='DarkBlue')

    