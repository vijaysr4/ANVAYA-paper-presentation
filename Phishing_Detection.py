# Importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



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

# Importing datasets
df = pd.read_csv("C:/Users/vijay/Downloads/csv_result.csv")

xx = df['Result'].value_counts().reset_index()
print(xx)


# Correlation Heatmap
plt.figure(figsize=(34, 12))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# Declaring dependent and independent variable
X = df.iloc[:,1:-1].values
y = df.iloc[:, -1].values

# Splitting datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 'mle')
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
'''
# ....Training Logistic Regression model..... 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting test results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Confusion matrix for Logistic Regression Classifier
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("``````````````````````````````````````````````````````\n")
print("Logistic Regression Classifier model\n")
print("Confusion Matrix for Logistic Regression Classifier model\n", cm)
print("\n\nAccuracy Score for Logistic Regression Classifier model\n", accuracy_score(y_test, y_pred))

confusion_matrix_visualization('Logistic Regression', 'Blues')

print("Train Score Logistic Regression:", classifier.score(X_train,y_train))
print("Test Score Logistic Reegression:", classifier.score(X_test,y_test))

# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("\nK-Fold Cross Validation\nAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# ....Training K-NN model.... 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting test results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), 1))

# Confusion matrix for K-NN Classifier
cm = confusion_matrix(y_test, y_pred)
print("``````````````````````````````````````````````````````\n")
print("K-NN Classifier model\n")
print("Confusion Matrix for K-NN Classifier model\n",cm)
print("\n\nAccuracy Score for K-NN Classifier model\n", accuracy_score(y_test, y_pred)) 

confusion_matrix_visualization('K-NN', 'cubehelix')

print("Train Score K-NN:", classifier.score(X_train,y_train))
print("Test Score K-NN:", classifier.score(X_test,y_test))

# K-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("\nK-Fold Cross Validation\nAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# .....Training SVM model.....
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting test results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), 1))

# Confusion matrix for SVM Classifier
cm = confusion_matrix(y_test, y_pred)
print("``````````````````````````````````````````````````````\n")
print("SVM Classifier model\n")
print("Confusion Matrix for SVM Classifier model\n",cm)
print("\n\nAccuracy Score for SVM Classifier model\n", accuracy_score(y_test, y_pred)) 

confusion_matrix_visualization('SVM', 'mako')

print("Train Score SVM:", classifier.score(X_train,y_train))
print("Test Score SVM:", classifier.score(X_test,y_test))

# K-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("\nK-Fold Cross Validation\nAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# .....Training Kernel SVM model.....
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting test results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), 1))

# Confusion matrix for Kernel SVM Classifier
cm = confusion_matrix(y_test, y_pred)
print("``````````````````````````````````````````````````````\n")
print("Kernel SVM Classifier model\n")
print("Confusion Matrix for Kernel SVM Classifier model\n",cm)
print("\n\nAccuracy Score for kernel SVM Classifier model\n", accuracy_score(y_test, y_pred)) 

confusion_matrix_visualization('Kernel SVM', 'Greys')

print("Train Score for Kernel SVM:", classifier.score(X_train,y_train))
print("Test Score for Kernel SVM:", classifier.score(X_test,y_test))

# K-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("\nK-Fold Cross Validation\nAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# .....Training Naive Bayes model..... 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#predicting test results
y_pred = classifier.predict(X_test)

# Confusion matrix for Naive Bayes Classifier
cm = confusion_matrix(y_test, y_pred)
print("``````````````````````````````````````````````````````\n")
print("Naive Bayes Classifier model\n")
print("Confusion Matrix for Naive Bayes Classifier model\n",cm)
print("\n\nAccuracy Score for Naive Bayes Classifier model\n", accuracy_score(y_test, y_pred))

confusion_matrix_visualization('Naive Bayes', 'magma')

print("Train Score for Naive Bayes:", classifier.score(X_train,y_train))
print("Test Score for Naive Bayes:", classifier.score(X_test,y_test))

# K-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("\nK-Fold Cross Validation\nAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# .....Training Decision Tree classifier  model..... 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#predicting test results
y_pred = classifier.predict(X_test)

# Confusion matrix for Decision Tree classifier
cm = confusion_matrix(y_test, y_pred)
print("``````````````````````````````````````````````````````\n")
print("Decision Tree classifier model\n")
print("Confusion Matrix for Decision Tree classifier model\n",cm)
print("\n\nAccuracy Score for Decision Tree classifier model\n", accuracy_score(y_test, y_pred))

confusion_matrix_visualization('Decision Tree classifier', 'rocket')

print("Train Score for Decision Tree:", classifier.score(X_train,y_train))
print("Test Score for Decision Tree:", classifier.score(X_test,y_test))

# K-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("\nK-Fold Cross Validation\nAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# .....Training Random Forest classifier  model.....
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#predicting test results
y_pred = classifier.predict(X_test)

# Confusion matrix for Random Forest classifier
cm = confusion_matrix(y_test, y_pred)
print("``````````````````````````````````````````````````````\n")
print("Random Forest Classifier Model\n")
print("Confusion Matrix for Random Forest classifier model\n",cm)
print("\n\nAccuracy Score for Random Forest classifier model\n", accuracy_score(y_test, y_pred))

confusion_matrix_visualization('Random Forest classifier', 'rocket')

print("Train Score for Random Forest classifier:", classifier.score(X_train,y_train))
print("Test Score for Random Forest classifier:", classifier.score(X_test,y_test))

# K-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("\nK-Fold Cross Validation\nAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Confusion matrix for XGBoost Classifier
cm = confusion_matrix(y_test, y_pred)
print("``````````````````````````````````````````````````````\n")
print("XGBoost Classifier model\n")
print("Confusion Matrix for XGBoost Classifier model\n",cm)
print("\n\nAccuracy Score for XGBoost Classifier model\n", accuracy_score(y_test, y_pred)) 

confusion_matrix_visualization('XGBoost', 'flare')

print("Train Score for XGBoost:", classifier.score(X_train,y_train))
print("Test Score for XGBoost:", classifier.score(X_test,y_test))

# K-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("\nK-Fold Cross Validation\nAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


#....Training Extra Tree Classifier
from sklearn.ensemble import ExtraTreesClassifier
classifier = ExtraTreesClassifier(n_estimators = 5, 
                           criterion = 'entropy',
                           max_features = 2)
classifier.fit(X_train, y_train)


# Confusion matrix for Extra Tree Classifier
cm = confusion_matrix(y_test, y_pred)
print("``````````````````````````````````````````````````````\n")
print("Extra Tree Classifier model\n")
print("Confusion Matrix for Extra Tree Classifier model\n",cm)
print("\n\nAccuracy Score for Extra Tree Classifier model\n", accuracy_score(y_test, y_pred)) 

confusion_matrix_visualization('Extra Tree', 'rocket')

print("Train Score for Extra Tree:", classifier.score(X_train,y_train))
print("Test Score for Extra Tree:", classifier.score(X_test,y_test))

# K-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("\nK-Fold Cross Validation\nAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Ada Boost Classifier
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1),
                                n_estimators = 200)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Confusion matrix for AdaBoost Classifier
cm = confusion_matrix(y_test, y_pred)
print("``````````````````````````````````````````````````````\n")
print("AdaBoost Classifier model\n")
print("Confusion Matrix for AdaBoost Classifier model\n",cm)
print("\n\nAccuracy Score for AdaBoost Classifier model\n", accuracy_score(y_test, y_pred)) 

confusion_matrix_visualization('AdaBoost', 'vlag')

print("Train Score for AdaBoost:", classifier.score(X_train,y_train))
print("Test Score for AdaBoost:", classifier.score(X_test,y_test))

# K-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("\nK-Fold Cross Validation\nAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(learning_rate = 0.1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Confusion matrix for Gradient Boostingt Classifier
cm = confusion_matrix(y_test, y_pred)
print("``````````````````````````````````````````````````````\n")
print("Gradient Boosting Classifier model\n")
print("Confusion Matrix for Gradient Boosting Classifier model\n",cm)
print("\n\nAccuracy Score for Gradient Boosting Classifier model\n", accuracy_score(y_test, y_pred)) 

confusion_matrix_visualization('Gradient Boosting', 'Spectral')

print("Train Score for Gradient Boosting:", classifier.score(X_train,y_train))
print("Test Score for Gradient Boosting:", classifier.score(X_test,y_test))

# K-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("\nK-Fold Cross Validation\nAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Light Gradient Boosting Machine Classifier
from lightgbm import LGBMClassifier
classifier = LGBMClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Confusion matrix for Light Gradient Boosting Machine Classifier
cm = confusion_matrix(y_test, y_pred)
print("``````````````````````````````````````````````````````\n")
print("Light Gradient Boosting Machine Classifier model\n")
print("Confusion Matrix for Light Gradient Boosting Machine Classifier model\n",cm)
print("\n\nAccuracy Score for Light Gradient Boosting Machine Classifier model\n", accuracy_score(y_test, y_pred)) 

confusion_matrix_visualization('Light Gradient Boosting Machine', 'mako')

print("Train Score for Light Gradient Boosting Machine:", classifier.score(X_train,y_train))
print("Test Score for Light Gradient Boosting Machine:", classifier.score(X_test,y_test))

# K-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("\nK-Fold Cross Validation\nAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



