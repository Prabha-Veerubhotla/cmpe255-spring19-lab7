import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from sklearn import svm, datasets, preprocessing


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

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
      
def polynomial_kernel(iris):
    # TODO
    # NOTE: use 8-degree in the degree hyperparameter. 
    # Trains, predicts and evaluates the model

    print("POLYNOMIAL KERNEL")
    # see the data
    print("iris data shape: {}".format(iris.shape) )

    # see head
    print("iris data head")
    print(iris.head())
    
    # process
    x1 = iris['sepal-length']
    x2 = iris['sepal-width']

    # Take the first two features. We could avoid this by using a two-dim dataset
    X=np.array(list(zip(x1,x2)), dtype = float)
    print(X.shape)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris['Class'])

    # split the data into train and test data
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # train the SVM
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel="poly", degree = 8,  C=0.025)  
    svclassifier.fit(X_train, y_train)  

     # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix  
    print("CONFUSION MATRIX FOR POLYNOMIAL KERNEL")
    print(confusion_matrix(y_test,y_pred))  
    print("CLASSIFICATION REPORT FOR POLYNOMIAL KERNEL")
    print(classification_report(y_test,y_pred))
    X0, X1 = x1, x2
    xx, yy = make_meshgrid(X0, X1)

    fig, ax = plt.subplots(1, 1)
    title = 'SVC with Polynomial (degree 8) kernel'
    clf = svclassifier
    plot_contours(ax, clf, xx, yy,
                cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

    plt.show()

def gaussian_kernel(iris):
    # TODO
    # Trains, predicts and evaluates the model

    print("GAUSSIAN KERNEL")
    # see the data
    print("iris data shape: {}".format(iris.shape) )
    
     # process
    x1 = iris['sepal-length']
    x2 = iris['sepal-width']

    # Take the first two features. We could avoid this by using a two-dim dataset
    X=np.array(list(zip(x1,x2)), dtype = float)
    print(X.shape)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris['Class'])

    # split the data into train and test data
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # train the SVM
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel="rbf",gamma=2, C = 1)  
    svclassifier.fit(X_train, y_train)  

     # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix  
    print("CONFUSION MATRIX FOR GAUSSIAN KERNEL")
    print(confusion_matrix(y_test,y_pred))  
    print("CLASSIFICATION REPORT FOR GAUSSIAN KERNEL")
    print(classification_report(y_test,y_pred))

    X0, X1 = x1, x2
    xx, yy = make_meshgrid(X0, X1)

    fig,ax = plt.subplots(1, 1)
    title = 'SVC with Guassian kernel'
    clf = svclassifier
    plot_contours(ax, clf, xx, yy,
                cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

    plt.show()

def sigmoid_kernel(iris):
    # TODO
    # Trains, predicts and evaluates the model

    print("SIGMOID KERNEL")
    # see the data
    print("iris data shape: {}".format(iris.shape) )
    
     # process
    x1 = iris['sepal-length']
    x2 = iris['sepal-width']

    # Take the first two features. We could avoid this by using a two-dim dataset
    X=np.array(list(zip(x1,x2)), dtype = float)
    print(X.shape)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris['Class'])


    # split the data into train and test data
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # train the SVM
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel="sigmoid", gamma= 2)  
    svclassifier.fit(X_train, y_train)  

     # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix  
    print("CONFUSION MATRIX FOR SIGMOID KERNEL")
    print(confusion_matrix(y_test,y_pred))  
    print("CLASSIFICATION REPORT FOR SIGMOID KERNEL")
    print(classification_report(y_test,y_pred)) 

    X0, X1 = x1, x2
    xx, yy = make_meshgrid(X0, X1)
    
    fig,ax = plt.subplots(1, 1)
    title = 'SVC with Sigmoid kernel'
    clf = svclassifier
    # for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

    plt.show()
    

def test():
    irisdata = import_iris()
    polynomial_kernel(irisdata)
    gaussian_kernel(irisdata)
    sigmoid_kernel(irisdata)
    # NOTE: 3-point extra credit for plotting three kernel models.

test()