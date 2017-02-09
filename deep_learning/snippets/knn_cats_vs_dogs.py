import argparse
import os
import sys
from operator import itemgetter

import numpy as np
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


def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


if __name__ == "__main__":
    # Load opencv libraries
    sys.path.append('/usr/local/lib/python2.7/site-packages')
    import cv2
    import imutils
    from imutils import paths

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
    ap.add_argument("-j", "--jobs", type=int, default=-1,
                    help="# of jobs for k-NN distance (-1 uses all available cores)")
    args = vars(ap.parse_args())

    # grab the list of images that we'll be describing
    print("[INFO] describing images...")
    imagePaths = list(paths.list_images(args["dataset"]))

    # initialize the raw pixel intensities matrix, the features matrix,
    # and labels list
    rawImages = []
    features = []
    labels = []

    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]

        # extract raw pixel intensity "features", followed by a color
        # histogram to characterize the color distribution of the pixels
        # in the image
        pixels = image_to_feature_vector(image)
        hist = extract_color_histogram(image)

        # update the raw images, features, and labels matricies,
        # respectively
        rawImages.append(pixels)
        features.append(hist)
        labels.append(label)

        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))

    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    rawImages = np.array(rawImages)
    features = np.array(features)
    labels = np.array(labels)
    print("[INFO] pixels matrix: {:.2f}MB".format(rawImages.nbytes / (1024 * 1000.0)))
    print("[INFO] features matrix: {:.2f}MB".format(features.nbytes / (1024 * 1000.0)))

    # partition the data into training and testing splits, using 75%
    # of the data for training and the remaining 25% for testing
    # (trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)

    # # train and evaluate a k-NN classifer on the raw pixel intensities
    # print("[INFO] evaluating raw pixel accuracy...")
    # model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
    # model.fit(trainRI, trainRL)
    # acc = model.score(testRI, testRL)
    # print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
    #
    # # train and evaluate a k-NN classifer on the histogram
    # # representations
    # print("[INFO] evaluating histogram accuracy...")
    # model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
    # model.fit(trainFeat, trainLabels)
    # acc = model.score(testFeat, testLabels)
    # print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

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
        clf.fit(trainFeat, trainLabels)
        score = clf.score(testFeat, testLabels)
        results[name] = score

    print "---------------------------"
    print "Evaluation results"
    print "---------------------------"

    # sorting results and print out
    sorted(results.items(), key=itemgetter(1))
    for name in results:
        print name + " accuracy: %0.3f" % results[name]
