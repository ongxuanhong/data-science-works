import glob
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from pydicom import dicomio

if __name__ == "__main__":
    img = mpimg.imread("imgs/dogs.jpg")
    print "Image data", img.shape
    print img

    print "Show image"
    plt.style.use("ggplot")
    plt.imshow(img)
    plt.colorbar()
    plt.show()

    print "Show RGB channels"
    plt.imshow(img[:, :, 0], cmap="gray")
    plt.show()
    plt.imshow(img[:, :, 1], cmap="gray")
    plt.show()
    plt.imshow(img[:, :, 2], cmap="gray")
    plt.show()

    root_dir = "sample_images/00cba091fa4ad62cc3200a657aeb957e/"
    os.chdir(root_dir)
    images = []
    for f in glob.glob("*.dcm"):
        ds = dicomio.read_file(f)
        img = ds.pixel_array
        images.append(img)

    # convert to array
    data = np.array(images)
    print "Total images:", len(images)
    print "Image dimensions:", images[0].shape
    print "Combine dimensions:", data.shape

    plt.style.use("ggplot")
    print "Calculating mean images"
    mean_img = np.mean(data, axis=0)
    plt.imshow(mean_img.astype(np.uint8))
    plt.show()

    print "Calculating deviation images"
    std_img = np.std(data, axis=0)
    plt.imshow(std_img.astype(np.uint8))
    plt.show()

    # convert to flattened array
    flattened = data.ravel()
    print "First image:", data[:1]
    print "First 10 values:", flattened[:10]

    print "Histogram"
    plt.hist(flattened, 255)
    plt.show()

    print "Histogram Equalization"
    plt.hist(mean_img.ravel(), 255)
    plt.show()

    print "Normalizing our data"
    bins = 20
    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)
    axs[0].hist(data[0].ravel(), bins)
    axs[0].set_title("img distribution")
    axs[1].hist(mean_img.ravel(), bins)
    axs[1].set_title("mean distribution")
    axs[2].hist((data[0] - mean_img).ravel(), bins)
    axs[2].set_title("(img - mean) distribution")
    plt.show()
