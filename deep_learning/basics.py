import matplotlib.image as mpimg
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img = mpimg.imread("imgs/dogs.jpg")
    print "Image data", img.shape
    print img

    print "Show image"
    plt.style.use('ggplot')
    plt.imshow(img)
    plt.show()

    print "Show RGB channels"
    plt.imshow(img[:, :, 0], cmap='gray')
    plt.show()
    plt.imshow(img[:, :, 1], cmap='gray')
    plt.show()
    plt.imshow(img[:, :, 2], cmap='gray')
    plt.show()