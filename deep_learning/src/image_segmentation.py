import argparse
import datetime
import glob
import os
import sys
import time

import numpy as np

sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import dicom as dicomio


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

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-s", "--saveto", required=True, help="path to saved processed data")
    args = vars(ap.parse_args())

    list_dir = os.listdir(args["dataset"])
    for (i, dir) in enumerate(list_dir):
        if os.path.isfile(dir) is False:
            basePath = args["dataset"] + "/" + dir
            os.chdir(basePath)
            images = []
            for f in glob.glob("*.dcm"):
                # read dcm file
                ds = dicomio.read_file(f)
                img = ds.pixel_array

                # normalize image values to [0, 255]
                cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
                img = cv2.medianBlur(img.astype(np.uint8), 5)

                # image segmentation
                thresh = cv2.adaptiveThreshold(img,
                                               255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY,
                                               11,
                                               2)
                images.append(thresh)

            data = np.array(images)
            mean_img = np.mean(data, axis=0)
            save_name = args["saveto"] + "/" + dir + ".png"
            cv2.imwrite(save_name, mean_img)
            print "Saved processed image:", save_name

        # show an update every 10 patients
        if i > 0 and i % 10 == 0:
            print "[INFO] processed {}/{} patients".format(i, len(list_dir))
            print "[INFO] time passed", time_diff_str(t_start, time.time())

    print "[INFO]", datetime.datetime.now(), "* DONE After *", time_diff_str(t_start, time.time())
