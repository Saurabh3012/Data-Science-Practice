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



#### Part 2: Data Modelling ####

# Importing the libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
# We will use rectifier function for hidden layers and sigmoid for output layer

# units - taken to be the (sum of number of(inputs + outputs))/2
# kernel_initializer - assigns input value to weights.
# activation - linear, sigmoid, rectifier, tanh 
# input_dim - to be entered just for the first time
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu', input_dim = 11))

# Adding a second input layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))

# Adding the output layer
# units - set to 1 because we just need 1 output
# Note - for multiple classes(>2) use units = no_of_classes and activation = 'softmax' 
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))

# Compiling the ANN
# loss = categorical_crossentropy - for classes > 2 (Non-binary)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

# Fitting the ANN to the training set.
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)



#### Part 3: Making the predictions and evaluating the model ####

# Predicting the test set results
y_pred = classifier.predict(X_test)
 
# To compare with the y_test (actual) we have to first convert these probabilities
# into true/false or 1/0
y_pred = (y_pred>0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Predicting for a single observation - To do this we have to encode our new inputs
# and scale them as well. 
'''Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000'''

single_ip = sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
y_single_pred = (classifier.predict(single_ip)>0.5)




















