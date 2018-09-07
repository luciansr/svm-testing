from sklearn import svm, datasets
import pandas
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import sys

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
            'name': 'Linear',
            'learner': svm.SVC(kernel='linear', C=C),
            'E_out': None,
            'E_out_description': None
        },
        {
            'name': 'Polinomial grau 3',
            'learner': svm.SVC(kernel='poly', degree=3, C=C),
            'E_out': None,
            'E_out_description': None
        },
        {
            'name': 'Polinomial grau 4',
            'learner': svm.SVC(kernel='poly', degree=4, C=C),
            'E_out': None,
            'E_out_description': None
        },
        {
            'name': 'Sigmoide gama 1',
            'learner': svm.SVC(kernel='sigmoid', gamma=1, C=C),
            'E_out': None,
            'E_out_description': None
        },
        {
            'name': 'Sigmoide gama 0.5',
            'learner': svm.SVC(kernel='sigmoid', gamma=0.5, C=C),
            'E_out': None,
            'E_out_description': None
        },
        {
            'name': 'Sigmoide gama 0.01',
            'learner': svm.SVC(kernel='sigmoid', gamma=0.01, C=C),
            'E_out': None,
            'E_out_description': None
        },
        {
            'name': 'RBF',
            'learner': svm.SVC(kernel='rbf', C=C),
            'E_out': None,
            'E_out_description': None
        }
    ]

    ## get data
    data = pandas.read_csv('./dataset/'+ datasetName)

    #print(svm_learner.predict([X[0, :]]))
    #print(X)
    #print(Y)

    row_number = data.values.shape[0]

    results_dic = dict()

    for k in k_folds:
        shuffledData = shuffle(data)
        X = shuffledData.values[:, :2]
        Y = shuffledData.values[:,2]

        for i in range(k): 
            k_range = (row_number+1)/k
            fold_range = [int(k_range * i), int(k_range*(i+1))]

            fold_test_data_X = X[fold_range[0]: fold_range[1], :]
            fold_test_data_Y = Y[fold_range[0]: fold_range[1]]

            X_trainning_part1 = X[0: fold_range[0], :]
            X_trainning_part2 = X[fold_range[1]: row_number, :]

            Y_trainning_part1 = Y[0: fold_range[0]]
            Y_trainning_part2 = Y[fold_range[1]: row_number]

            fold_trainning_data_X = np.concatenate((X_trainning_part1, X_trainning_part2))   
            fold_trainning_data_Y = np.append(Y_trainning_part1, Y_trainning_part2)   
            #print(fold_test_data_X.shape[0] + fold_trainning_data_X.shape[0])
            for item in typesOfSVMs:
                clf_name = item['name']
                clf = item['learner']

                result_key_name = clf_name + '_k_fold_' + str(k)

                if not(result_key_name in results_dic):
                    results_dic[result_key_name] = {
                        'E_out': None,
                        'E_out_description': None
                    }

                dict_item = results_dic[result_key_name]

                clf.fit(fold_trainning_data_X, fold_trainning_data_Y)
                clf_prediction = clf.predict(fold_test_data_X)

                test_row_number = fold_test_data_X.shape[0]
                errorsInPrediction = 0

                for prediction, real_test_Value in zip(clf_prediction, fold_test_data_Y):
                    if(prediction != real_test_Value):
                        errorsInPrediction += 1

                E_out = errorsInPrediction/test_row_number

                if(dict_item['E_out'] == None or dict_item['E_out'] > E_out):
                    dict_item['E_out'] = E_out
                    dict_item['E_out_description'] = 'Min E_out = ' + str(E_out) + ' on k_fold ' + str(k)

                if(item['E_out'] == None or item['E_out'] > E_out):
                    item['E_out'] = E_out
                    item['E_out_description'] = 'Min E_out = ' + str(E_out) + ' on k_fold ' + str(k)
    ## end testing

    for key,value in results_dic.items():
        print(key + ': ' + value['E_out_description'])

    print('\nfinal result:')

    for item in typesOfSVMs:
        print(item['name']+ ': ' + item['E_out_description'])
  

    return None

test_svm('banana-data.dat')