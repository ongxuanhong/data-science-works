import argparse
import csv
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


def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


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


def get_simple_feature_labels(patient_labels, img_paths):
    features = []
    labels = []

    # loop over the input images
    for (i, img_path) in enumerate(img_paths):
        # get only training labels
        base = os.path.basename(img_path)
        patient_id = os.path.splitext(base)[0]
        if patient_id in patient_labels["cancer"].keys():
            labels.append(patient_labels["cancer"][patient_id])
        else:
            continue

        # load the image
        image = cv2.imread(img_path)

        # histogram to characterize the color distribution of the pixels
        # in the image
        feat = image_to_feature_vector(image)

        # update features
        features.append(feat)

        # show an update every 100 images
        if i > 0 and i % 100 == 0:
            print("[INFO] processed {}/{}".format(i, len(img_paths)))

    return features, labels


def get_hist_feature_labels(patient_labels, img_paths):
    features = []
    labels = []

    # loop over the input images
    for (i, img_path) in enumerate(img_paths):
        # get only training labels
        base = os.path.basename(img_path)
        patient_id = os.path.splitext(base)[0]
        if patient_id in patient_labels["cancer"].keys():
            labels.append(patient_labels["cancer"][patient_id])
        else:
            continue

        # load the image
        image = cv2.imread(img_path)

        # histogram to characterize the color distribution of the pixels
        # in the image
        hist = extract_color_histogram(image)

        # update features
        features.append(hist)

        # show an update every 100 images
        if i > 0 and i % 100 == 0:
            print("[INFO] processed {}/{}".format(i, len(img_paths)))

    return features, labels


def generate_bow_features(img_paths, dictionarySize=5):
    BOW = cv2.BOWKMeansTrainer(dictionarySize)
    sift = cv2.xfeatures2d.SIFT_create()

    for (i, image_path) in enumerate(img_paths):
        gray = cv2.imread(image_path)
        kp, dsc = sift.detectAndCompute(gray, None)
        BOW.add(dsc)
        print("# kps: {}, descriptors: {}".format(len(kp), dsc.shape))

    # dictionary created
    dictionary = BOW.cluster()
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    sift2 = cv2.xfeatures2d.SIFT_create()
    bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
    bowDiction.setVocabulary(dictionary)
    print "[INFO] Finished create BOW dictionary", time_diff_str(t_start, time.time())
    return bowDiction


def sift_feature_extract(img_paths, patient_labels, bow_dict):
    features = []
    labels = []

    # loop over the input images
    for (i, img_path) in enumerate(img_paths):
        # get only training labels
        base = os.path.basename(img_path)
        patient_id = os.path.splitext(base)[0]
        if patient_id in patient_labels["cancer"]:
            labels.append(patient_labels["cancer"][patient_id])
        else:
            continue

        # load the image
        gray = cv2.imread(img_path)
        sift_feature = bow_dict.compute(gray, sift.detect(gray))

        # update features
        features.extend(sift_feature)

        # show an update every 100 images
        if i > 0 and i % 100 == 0:
            print("[INFO] processed {}/{}".format(i, len(img_paths)))

    return features, labels


if __name__ == "__main__":
    t_start = time.time()

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs (-1 uses all available cores)")
    args = vars(ap.parse_args())

    # grab the list of images that we'll be describing
    print("[INFO] describing images...")
    img_paths = list(paths.list_images(args["dataset"]))

    # load train/test labels
    stage1_labels = load_csv("../data/stage1_labels.csv")
    stage1_sample_submission = load_csv("../data/stage1_sample_submission.csv")

    # Generating Bag of Words model
    # generate_bow_features(img_paths)

    # train_features, train_labels = get_hist_feature_labels(stage1_labels, img_paths)
    # test_features, test_labels = get_hist_feature_labels(stage1_sample_submission, img_paths)
    # train_features, train_labels = sift_feature_extract(img_paths, stage1_labels, bowDiction)
    # test_features, test_labels = sift_feature_extract(img_paths, stage1_sample_submission, bowDiction)
    train_features, train_labels = get_simple_feature_labels(stage1_labels, img_paths)
    test_features, test_labels = get_simple_feature_labels(stage1_sample_submission, img_paths)
    train_features = np.array(train_features)
    print("[INFO] features matrix: {:.2f}MB".format(train_features.nbytes / (1024 * 1000.0)))

    (for_train_features, dev_features, for_train_labels, dev_labels) = train_test_split(train_features,
                                                                                        train_labels,
                                                                                        test_size=0.25,
                                                                                        random_state=42)

    print "---------------------------"
    print "Training"
    print "---------------------------"

    classifiers = {
        "Nearest Neighbors": KNeighborsClassifier(3, n_jobs=args["jobs"]),
        "Linear SVM": SVC(kernel="linear", C=0.025),
        "RBF SVM": SVC(gamma=2, C=1),
        "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True, n_jobs=args["jobs"]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=args["jobs"]),
        "Neural Net": MLPClassifier(alpha=1),
        "AdaBoost": AdaBoostClassifier(),
        "Naive Bayes": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis()
    }

    # iterate over classifiers
    results = {}

    for name in classifiers:
        print "[INFO]" + name + " classifier..."
        clf = classifiers[name]
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

    print "---------------------------"
    print "Training for submission"
    print "---------------------------"

    name = list(results)[0]
    clf = classifiers[name]
    print "[INFO]" + name + " classifier..."
    clf.fit(train_features, train_labels)
    predict_submission = clf.predict(test_features)

    # update submission
    submission = {}
    for (i, patient_id) in enumerate(stage1_sample_submission["cancer"]):
        submission[patient_id] = predict_submission[i]

    with open("submission_results.csv", "wb") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["id", "cancer"])
        for key, value in submission.items():
            writer.writerow([key, value])

    print "[INFO]", datetime.datetime.now(), "* DONE After *", time_diff_str(t_start, time.time())
