from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
import pandas
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

## parameters
# datasetName = filename
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

def plot(clf, title, ax, xx, yy, X0, X1, y, plt):
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
    
    return 

def test_svm(datasetName):
    

    ## k folds
    k_folds = [2,5,10]

    ## setup all svms
    C = 1.0  # SVM regularization parameter
    typesOfKNNs = [
        {
            'name': '2 Vizinhos',
            'learner': KNeighborsClassifier(n_neighbors=2)
        },
        {
            'name': '3 Vizinhos',
            'learner': KNeighborsClassifier(n_neighbors=3)
        },
        {
            'name': '5 Vizinhos',
            'learner': KNeighborsClassifier(n_neighbors=5)
        },
        {
            'name': '10 Vizinhos',
            'learner': KNeighborsClassifier(n_neighbors=10)
        }
    ]

    ## get data
    data = pandas.read_csv('./dataset/'+ datasetName)

    # X = data.values[:, :4]
    X = data.values[:, :2]
    Y = data.values[:,2]

    svm_learner = svm.SVC(kernel='poly', degree=3)
    svm_learner.fit(X, Y)

    typesOfKNNs = ({ 
        'name': item['name'],
        'learner': item['learner'].fit(X, Y) 
    } for item in typesOfKNNs)

    typesOfKNNs = ({ 
        'name': item['name'],
        'learner': item['learner'],
        'resultY': item['learner'].predict(X) # if item['name'] != 'Original Data' else Y
    } for item in typesOfKNNs)

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 4)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for SVMitem, ax in zip(typesOfKNNs, sub.flatten()):
        clf = SVMitem['learner']
        title = SVMitem['name']
        resultY = SVMitem['resultY']
        plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=resultY, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('At1')
        ax.set_ylabel('At2')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    
    plt.show()

    return None

test_svm('banana-data.dat')