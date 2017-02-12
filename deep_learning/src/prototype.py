import argparse
import glob
import os
import sys

sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import dicom as dicomio
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np


def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    args = vars(ap.parse_args())

    os.chdir(args["dataset"])
    images = []
    for f in glob.glob("*.dcm"):
        # read dcm file
        ds = dicomio.read_file(f)
        img = ds.pixel_array
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        pixels = image_to_feature_vector(img)
        images.append(pixels)

    plt.imshow(images)
    plt.show()

    import pandas as pd

    df = pd.DataFrame(np.array(images))
    df.to_csv("hello.csv")
