import argparse
import datetime
import os
import sys
import time
from operator import itemgetter

import numpy as np
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

sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import imutils
from imutils import paths


def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"


def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0], None, [8], [0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


def load_csv(file_path):
    """Get data, from local csv."""
    if os.path.exists(file_path):
        print "[INFO] load", file_path, "file..."
        df = pd.read_csv(file_path, index_col=0)

    return df.to_dict()


if __name__ == "__main__":
    t_start = time.time()

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs (-1 uses all available cores)")
    args = vars(ap.parse_args())

    # grab the list of images that we'll be describing
    print("[INFO] describing images...")
    imagePaths = list(paths.list_images(args["dataset"]))

    # load train/test labels
    stage1_labels = load_csv("stage1_labels.csv")
    stage1_sample_submission = load_csv("stage1_sample_submission.csv")

    # initialize features matrix and labels list
    train_features = []
    train_labels = []

    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
        # get only training labels
        base = os.path.basename(imagePath)
        patient_id = os.path.splitext(base)[0]
        if patient_id in stage1_labels["cancer"].keys():
            train_labels.append(stage1_labels["cancer"][patient_id])
        else:
            continue

        # load the image
        image = cv2.imread(imagePath)

        # histogram to characterize the color distribution of the pixels
        # in the image
        hist = extract_color_histogram(image)

        # update features
        train_features.append(hist)

        # show an update every 100 images
        if i > 0 and i % 100 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))

    train_features = np.array(train_features)
    print("[INFO] features matrix: {:.2f}MB".format(train_features.nbytes / (1024 * 1000.0)))

    (for_train_features, dev_features, for_train_labels, dev_labels) = train_test_split(train_features,
                                                                                        train_labels,
                                                                                        test_size=0.25,
                                                                                        random_state=42)

    print "---------------------------"
    print "Training"
    print "---------------------------"

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3, n_jobs=args["jobs"]),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True, n_jobs=args["jobs"]),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=args["jobs"]),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    # iterate over classifiers
    results = {}

    for name, clf in zip(names, classifiers):
        print "Training " + name + " classifier..."
        clf.fit(for_train_features, for_train_labels)
        score = clf.score(dev_features, dev_labels)
        results[name] = score

    print "---------------------------"
    print "Evaluation results"
    print "---------------------------"

    # sorting results and print out
    sorted(results.items(), key=itemgetter(1))
    for name in results:
        print "[INFO]", name, "accuracy: %0.3f" % results[name]

    print "[INFO]", datetime.datetime.now(), "* DONE After *", time_diff_str(t_start, time.time())
