import argparse
import datetime
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# returns descriptor of image at pth
def feature_extract(pth):
    im = cv2.imread(pth, 1)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return bowDiction.compute(gray, sift.detect(gray))


def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"


if __name__ == "__main__":
    # Load opencv libraries
    sys.path.append('/usr/local/lib/python2.7/site-packages')
    import cv2
    from imutils import paths

    t_start = time.time()

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
    features = []
    labels = []

    dictionarySize = 5
    BOW = cv2.BOWKMeansTrainer(dictionarySize)
    sift = cv2.xfeatures2d.SIFT_create()

    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    print "BOW dictionary", np.shape(dictionary)

    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        label = imagePath.split(os.path.sep)[-1].split(".")[0]

        # update the raw images, features, and labels matricies,
        # respectively
        features.extend(feature_extract(imagePath))
        labels.append(label)

        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))

    # show some information on the memory consumed by the features matrix
    features = np.array(features)
    labels = np.array(labels)
    print("[INFO] features matrix: {:.2f}MB".format(features.nbytes / (1024 * 1000.0)))

    # partition the data into training and testing splits, using 75%
    # of the data for training and the remaining 25% for testing
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)

    # train and evaluate a k-NN classifer on the histogram
    # representations
    print("[INFO] evaluating accuracy...")
    model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
    model.fit(trainFeat, trainLabels)
    acc = model.score(testFeat, testLabels)
    print("[INFO] accuracy: {:.2f}%".format(acc * 100))

    print "-- %s * DONE After * %s" % (datetime.datetime.now(), time_diff_str(t_start, time.time()))
