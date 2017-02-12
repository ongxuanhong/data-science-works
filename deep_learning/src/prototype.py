import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
from imutils import paths
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def load_csv(file_path):
    """Get data, from local csv."""
    if os.path.exists(file_path):
        print "[INFO] load", file_path, "file..."
        df = pd.read_csv(file_path)

    return df


def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def get_simple_feature_labels(patient_df, img_paths):
    features = []
    labels = []

    patient_ids = patient_df["id"].tolist()

    # loop over the input images
    for (i, img_path) in enumerate(img_paths):
        # get only training labels
        base = os.path.basename(img_path)
        patient_id = os.path.splitext(base)[0]
        if patient_id in patient_ids:
            label = patient_df[patient_df["id"] == patient_id].iloc[0]["cancer"]
            labels.append(label)
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


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-s", "--save", help="path to save features")
    args = vars(ap.parse_args())

    # load train/test labels
    stage1_labels = load_csv("../data/stage1_labels.csv")
    stage1_sample_submission = load_csv("../data/stage1_sample_submission.csv")

    img_paths = list(paths.list_images(args["dataset"]))
    train_features, train_labels = get_simple_feature_labels(stage1_labels, img_paths)

    train_labels = np.array(train_labels)
    train_features = np.array(train_features)
    print "[INFO] labels vector shape:", train_labels.shape
    print "[INFO] features matrix shape:", train_features.shape
    print("[INFO] features matrix size: {:.2f}MB".format(train_features.nbytes / (1024 * 1000.0)))

    plt.imshow(train_features)
    plt.show()
