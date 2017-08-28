# -*- coding: utf-8 -*-

#### Part 1: Data Pre processing ####

# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Datasets.
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13 ].values

# Encoding the categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encoding Geography
labelencoder_X_geography = LabelEncoder()
X[:,1] = labelencoder_X_geography.fit_transform(X[:,1])

# Encoding Gender
labelencoder_X_gender = LabelEncoder()
X[:,2] = labelencoder_X_gender.fit_transform(X[:,2]) 

# In cases where there are more than one categories, let say our encoder
# made them 0, 1 and 2. Then the system will assume 2 to be bigger than 
# the 0 or 1. There we convert such cases to using OneHotEncoder to avoid
# this.

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Drop one of the columns created using prev step to avoid the dummy
# variable trap
X = X[:, 1:] 

# Splitting the data set into Train,Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# Feature Scaling - when the difference between the ranges of two different
# independent variables is too large, we scale them so each lies in the same
# region. Eases out math operations and solves the problem of one independent
# variable dominating over another.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test) 


'''in continuation to ann.py'''
#### Part 4: Evaluating the model####

#Evaluating the ANN

# Imports
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return classifier

# classifier trained with k-fold cross-validation
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

# Running 10 fold cross-validation
accuracies = cross_val_score(classifier, X_train, y_train, cv = 10, n_jobs = -1 )

mean = accuracies.mean()
variance = accuracies.std()



















