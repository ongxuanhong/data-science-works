# python3 color_kmeans.py --image son_tung.png --clusters 5
import argparse
import os

import cv2
import numpy as np
from sklearn.cluster import KMeans


def get_color_palette(k_cluster, centroids, palette_w=600, palette_h=100):
    # initialize the color palette
    text_y = int(palette_h / 2)
    palette = np.zeros((palette_h, palette_w, 3), dtype="uint8")
    startX = 0

    # loop over the color of each cluster
    for color in centroids:
        # plot the relative percentage of each cluster
        endX = startX + (1.0 / k_cluster * palette_w)
        text_x = int(startX + 15)

        bgr_code = str(color.astype("uint8").tolist()[0]) + ","
        bgr_code += str(color.astype("uint8").tolist()[1]) + ","
        bgr_code += str(color.astype("uint8").tolist()[2])

        cv2.rectangle(palette, (int(startX), 0), (int(endX), palette_h), color.astype("uint8").tolist(), -1)
        cv2.putText(palette, bgr_code, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200))
        startX = endX

    # return the palette
    return palette


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-c", "--clusters", required=True, type=int, help="# of clusters")
    args = vars(ap.parse_args())

    for f in os.listdir(args["image"]):
        if f.endswith(".png"):
            img_path = args["image"] + "/" + f
            img_name = os.path.splitext(f)[0]

            # load the image
            image = cv2.imread(img_path)

            # reshape the image to be a list of pixels
            image = image.reshape((image.shape[0] * image.shape[1], 3))

            # cluster the pixel intensities
            clt = KMeans(n_clusters=args["clusters"])
            clt.fit(image)

            # representing the number of pixels labeled to each color
            palette = get_color_palette(args["clusters"], clt.cluster_centers_)

            # save color palette
            fig_out = "fig_out/color_pallete_" + img_name + ".png"
            cv2.imwrite(fig_out, palette)
            print("Done", f)
