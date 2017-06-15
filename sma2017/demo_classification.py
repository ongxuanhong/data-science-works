import csv
import datetime
import itertools
import time
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":
    t_start = time.time()
    print "------------------------------------------------"
    print " %s - Doing classification " % datetime.datetime.now()
    print "------------------------------------------------"

    df_data = pd.read_csv("data/knxad_knx2991_201604204777.csv")
    features = list(df_data.columns[:4])
    class_names = [-1, 0, 1]

    y = df_data["decision"]
    X = df_data[features]

    train, test = train_test_split(df_data, test_size=0.2)
    X_train = train[features]
    y_train = train["decision"]

    X_test = test[features]
    y_test = test["decision"]

    print "---------------------------"
    print "Training"
    print "---------------------------"

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="sigmoid", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    # iterate over classifiers
    results = {}

    with open("data/classification_pcf_results.csv", "wb") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["classifiers", "precision", "recall", "fscore"])

        for name, clf in zip(names, classifiers):
            print "Training " + name + " classifier..."
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            results[name] = score

            # Compute confusion matrix
            y_pred = clf.predict(X_test)
            cnf_matrix = confusion_matrix(y_test, y_pred)
            np.set_printoptions(precision=2)

            # Plot normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                  title='Normalized confusion matrix')

            plt.savefig("figs/" + name)

            pcf = precision_recall_fscore_support(y_test, y_pred)
            print "P,C,F", pcf
            writer.writerow([name, pcf[0], pcf[1], pcf[2]])

    print "---------------------------"
    print "Evaluation results"
    print "---------------------------"

    # sorting results and print out
    sorted(results.items(), key=itemgetter(1))
    with open("data/classification_results.csv", "wb") as f:
        writer = csv.writer(f, delimiter=",")
        for name in results:
            print name + " accuracy: %0.3f" % results[name]
            writer.writerow([name, results[name]])

    print " %s * DONE After * %s" % (datetime.datetime.now(), time_diff_str(t_start, time.time()))
