import csv
import datetime
import time
from operator import itemgetter

import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
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


if __name__ == "__main__":
    t_start = time.time()
    print "------------------------------------------------"
    print " %s - Doing classification " % datetime.datetime.now()
    print "------------------------------------------------"

    df_data = pd.read_csv("data/knxad_knx2991_201604204777.csv")
    features = list(df_data.columns[:4])
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
        SVC(kernel="linear", C=0.025),
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

    for name, clf in zip(names, classifiers):
        print "Training " + name + " classifier..."
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        results[name] = score

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
