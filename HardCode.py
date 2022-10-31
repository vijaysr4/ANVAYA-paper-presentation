# Importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score

def confusion_matrix_visualization(model_name, color_seq):
    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]

    labels = [f"{v1}\n{v2}" for v1, v2, in
              zip(group_names, group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(cm, annot = labels, fmt = '' , cmap = color_seq)

    ax.set_title('Confusion Matrix for ' + model_name + ' Classifier')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')

    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    plt.show()

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def Euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class LogisticReg:
    
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters 
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        # Initialise parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        # Linear model
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            dw = (1 / n_samples) + np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) + np.sum(y_predicted - y)
            
            self.weights -= self.lr + dw
            self.bias -= self.lr + db
            
            
            
    def predict(self, X):
        # Linear model + Sigmoid function
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        

class KNN:
    def __init__(self, k = 3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # Compute Distance
        distances = [Euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get K nearest samples, labels
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
        
# Importing datasets
df = pd.read_csv('D:/Project_and_Case_Study_1/Final_shuffle.csv')

# Declaring dependent and independent variable
X = df.iloc[:1000, 2:-1].values
y = df.iloc[:1000, -1].values

# Splitting datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

regressor = LogisticReg(lr = 0.0001, n_iters = 1000)
regressor.fit(X_train, y_train)
train_predictions = regressor.predict(X_train)
predictions = regressor.predict(X_test)

print("Logistic Regressor Classification training accuracy: ", accuracy(y_train, train_predictions)) 
print("Logistic Regressor Classification accuracy: ", accuracy(y_test, predictions)) 


cm = confusion_matrix(y_test, predictions)
print("``````````````````````````````````````````````````````\n")
print("Logistic Regression Classifier model\n")
print("Confusion Matrix for Logistic Regression Classifier model\n", cm)
print("\n\nAccuracy Score for Logistic Regression Classifier model\n", accuracy_score(y_test, predictions))

confusion_matrix_visualization('Logistic Regression', 'Blues')



clf = KNN(k = 8)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)


# Confusion matrix for K-NN Classifier
cm = confusion_matrix(y_test, predictions)
print("``````````````````````````````````````````````````````\n")
print("K-NN Classifier model\n")
print("Confusion Matrix for K-NN Classifier model\n",cm)
print("\n\nAccuracy Score for K-NN Classifier model\n", accuracy_score(y_test, predictions)) 

confusion_matrix_visualization('K-NN', 'cubehelix')



class SVM:
    def __init__(self, learning_rate = 0.001, lambda_param = 0.01, n_iters = 1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        
        # Defining y for -1 and 1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Gradient desent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)


clf = SVM()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("SVM classification accuracy", accuracy(y_test, predictions))



class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("Naive Bayes classification accuracy", accuracy(y_test, predictions))

