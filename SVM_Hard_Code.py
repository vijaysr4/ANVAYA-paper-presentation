# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing Datasets
df = pd.read_csv('D:/Project_and_Case_Study_1/Final_shuffle.csv')
print(df.head())

# Declaring dependent and independent variable
X = df.iloc[:1000, 2:-1].values
y = df.iloc[:1000, -1].values

# Splitting datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

class SVM_classifier():
    
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations 
        self.lambda_parameter = lambda_parameter
        
        
    def fit(self, X, y):
        
        self.m, self.n = X.shape
        
        # Initiating weights and bias
        
        self.w = np.zero(self.n)
        
        self.b = 0
        self.X = X
        self.y = y
        
        # Implementing Gradient Descent
        
        for i in range(self.no_of_iterations):
            self.update_weights()
        
    def update_weights():
        # label encoding
        y_label = np.where(self.y <= 0, -1, 1)
        
        for index, x_i in enumerate(self.X):
            
            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1
            
            if (condition == True):
                
                dw = 2 * self.lambda_parameter * self.w
                db = 0
                
            else:
                
                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
                db = y_label[index]
            
    def predict(self, X):
        
        output = np.dot(X, self.w) - self.b
        
        predict_labels = np.sign(output)
        
        y_hat = np.where(predicted_labels <= -1, 0, 1)
        
        return y_hat()
        