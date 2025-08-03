# Importing dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Loading dataset
diabetes_dataset = pd.read_csv('Diabetes Predictor/diabetes.csv')  # Ensure the path is correct


# Displaying the first 5 rows of the dataset
diabetes_dataset.head()

# Number of rows and columns in the dataset
diabetes_dataset.shape

# Getting the statistical measures of the dataset
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

# Separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']
print(X)
print(y)

# Data Standarization
scalar = StandardScaler()
scalar.fit(X)
StandardScaler(copy=True, with_mean=True, with_std=True)
standardized_data = scalar.transform(X)
print(standardized_data)
X = standardized_data
y = diabetes_dataset['Outcome']
print(X)
print(y)

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)
print(X.shape, X_train.shape, X_test.shape)

#Training the model
classifier = svm.SVC(kernel='linear')

#Training the Support Vector Machine Classifier
classifier.fit(X_train, y_train)

#Model Evaluation
#Accuracy Score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print('Accuracy on training data : ', training_data_accuracy)  

#Accuracy Score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(y_test, X_test_prediction)
print('Accuracy on test data : ', test_data_accuracy)

# Making a Predictive System
input_data = (1,89,66,23,94,28.1,0.167,21)  # Example input data

# Changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1) 

# standardize the input data
std_data= scalar.transform(input_data_reshaped)
print(std_data)

# Making prediction
prediction = classifier.predict(std_data)
print(prediction)
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')