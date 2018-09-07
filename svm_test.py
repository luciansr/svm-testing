from sklearn import svm, datasets
import pandas
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# %matplotlib inline

## parameters
## datasetName = filename
## kernelType: 
#    The kernel function can be any of the following:
#    
#    linear: (x, x').
#    polynomial: (\gamma \langle x, x'\rangle + r)^d. d is specified by keyword degree, r by coef0.
#    rbf: \exp(-\gamma \|x-x'\|^2). \gamma is specified by keyword gamma, must be greater than 0.
#    sigmoid (\tanh(\gamma \langle x,x'\rangle + r)), where r is specified by coef0.

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
    typesOfSVMs = [
        {
            'name': 'linear',
            'learner': svm.SVC(kernel='linear', C=C)
        },
        {
            'name': 'polinomial grau 3',
            'learner': svm.SVC(kernel='poly', degree=3, C=C)
        },
        {
            'name': 'polinomial grau 4',
            'learner': svm.SVC(kernel='poly', degree=4, C=C)
        },
        {
            'name': 'sigmoide gama 1',
            'learner': svm.SVC(kernel='sigmoid', gamma=1, C=C)
        },
        {
            'name': 'sigmoide gama 0.5',
            'learner': svm.SVC(kernel='sigmoid', gamma=0.5, C=C)
        },
        {
            'name': 'sigmoide gama 0.01',
            'learner': svm.SVC(kernel='sigmoid', gamma=0.01, C=C)
        },
        {
            'name': 'rbf',
            'learner': svm.SVC(kernel='rbf', C=C)
        },
    ]

    ## get data
    data = pandas.read_csv('./dataset/'+ datasetName)

    X = data.values[:, :2]
    Y = data.values[:,2]

    svm_learner = svm.SVC(kernel='poly', degree=3)
    svm_learner.fit(X, Y)

    typesOfSVMs = ({ 
        'name': item['name'],
        'learner': item['learner'].fit(X, Y) 
    } for item in typesOfSVMs)

    typesOfSVMs = ({ 
        'name': item['name'],
        'learner': item['learner'],
        'resultY': item['learner'].predict(X)
    } for item in typesOfSVMs)

    #print(svm_learner.predict([X[0, :]]))
    #print(X)
    #print(Y)

    ## plot
    # X0, X1 = X[:, 0], X[:, 1]
    # xx, yy = make_meshgrid(X0, X1)

    # fig, sub = plt.subplots(1, 1)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # # plot(clf, title, ax, xx, yy, X0, X1, y, plt = plt):

    # models = (svm_learner, svm_learner)
    # # models = (clf.fit(X, y) for clf in models)

    # # title for the plots
    # titles = ('SVC with linear kernel','SVC with linear kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(3, 3)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    #plot(svm_learner, 'SVC with linear kernel', sub, xx, yy, X0, X1, Y, plt)


    # for clf, title, ax in zip(models, titles, sub):
    #     plot_contours(ax, clf, xx, yy,
    #               cmap=plt.cm.coolwarm, alpha=0.8)
    #     ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    #     ax.set_xlim(xx.min(), xx.max())
    #     ax.set_ylim(yy.min(), yy.max())
    #     ax.set_xlabel('Sepal length')
    #     ax.set_ylabel('Sepal width')
    #     ax.set_xticks(())
    #     ax.set_yticks(())
    #     ax.set_title(title)

    for SVMitem, ax in zip(typesOfSVMs, sub.flatten()):
        clf = SVMitem['learner']
        title = SVMitem['name']
        resultY = SVMitem['resultY']
        plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=resultY, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    
    plt.show()


    for item in typesOfSVMs:
        name = item['name']
        classifier = item['learner']
        print(name)

        for k in k_folds:
            shuffledData = shuffle(data)
            X = shuffledData.values[:, :2]
            Y = shuffledData.values[:,2]

            for i in range(k): 
                break




    E_out = 1

    return E_out

test_svm('banana-data.dat')