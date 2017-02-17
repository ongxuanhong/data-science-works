import argparse
import sys

# Load opencv libraries
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    cv2.imshow("image", image)
    cv2.waitKey(0)
