import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from sklearn import svm, datasets


def linear_svm():
    # download data set: https://drive.google.com/file/d/13nw-uRXPY8XIZQxKRNZ3yYlho-CYm_Qt/view
    # info: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

    # load data
    bankdata = pd.read_csv("/tmp/bill_authentication.csv")  

    # see the data
    bankdata.shape  

    # see head
    bankdata.head()  

    # data processing
    X = bankdata.drop('Class', axis=1)  
    y = bankdata['Class']  


    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # train the SVM
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(X_train, y_train)  

    # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix  
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))  


# Iris dataset  https://archive.ics.uci.edu/ml/datasets/iris4
def import_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    irisdata = pd.read_csv(url, names=colnames) 

    return irisdata

def polynomial_kernel(irisdata):
    # TODO
    # NOTE: use 8-degree in the degree hyperparameter. 
    # Trains, predicts and evaluates the model

    print("POLYNOMIAL KERNEL")
    # see the data
    print("iris data shape: {}".format(irisdata.shape) )

    # see head
    print("iris data head")
    print(irisdata.head())
    
     # process
    X = irisdata.drop('Class', axis=1)  
    y = irisdata['Class'] 

    # split the data into train and test data
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # train the SVM
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='poly', degree = 8)  
    svclassifier.fit(X_train, y_train)  

     # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix  
    print("CONFUSION MATRIX FOR POLYNOMIAL KERNEL")
    print(confusion_matrix(y_test,y_pred))  
    print("CLASSIFICATION REPORT FOR POLYNOMIAL KERNEL")
    print(classification_report(y_test,y_pred))

    # X = np.array(X)
    # y = np.array(y)
    # print(X.shape)
    # print(y.shape)

    # Z = svclassifier.predict(X)
    # Z = Z.reshape(X.shape)
    # print(Z.shape)


def gaussian_kernel(irisdata):
    # TODO
    # Trains, predicts and evaluates the model

    print("GAUSSIAN KERNEL")
    # see the data
    print("iris data shape: {}".format(irisdata.shape) )
    
     # process
    X = irisdata.drop('Class', axis=1)  
    y = irisdata['Class']  
    
    # split the data into train and test data
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # train the SVM
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='rbf', degree = 8)  
    svclassifier.fit(X_train, y_train)  

     # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix  
    print("CONFUSION MATRIX FOR GAUSSIAN KERNEL")
    print(confusion_matrix(y_test,y_pred))  
    print("CLASSIFICATION REPORT FOR GAUSSIAN KERNEL")
    print(classification_report(y_test,y_pred))

def sigmoid_kernel(irisdata):
    # TODO
    # Trains, predicts and evaluates the model

    print("SIGMOID KERNEL")
    # see the data
    print("iris data shape: {}".format(irisdata.shape) )
    
     # process
    X = irisdata.drop('Class', axis=1)  
    y = irisdata['Class']  
    
    # split the data into train and test data
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # train the SVM
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='sigmoid', degree = 8)  
    svclassifier.fit(X_train, y_train)  

     # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix  
    print("CONFUSION MATRIX FOR SIGMOID KERNEL")
    print(confusion_matrix(y_test,y_pred))  
    print("CLASSIFICATION REPORT FOR SIGMOID KERNEL")
    print(classification_report(y_test,y_pred)) 
    

def test():
    irisdata = import_iris()
    polynomial_kernel(irisdata)
    gaussian_kernel(irisdata)
    sigmoid_kernel(irisdata)
    # NOTE: 3-point extra credit for plotting three kernel models.

test()