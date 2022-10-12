from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


SIGMA = 10000


def create_filter(size):
    filter = np.zeros((size, size), np.float32)
    size = size // 2
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            x1 = 2 * np.pi * (SIGMA ** 2)
            x2 = np.exp(-(i**2 + j**2) / (2 * SIGMA ** 2))
            filter[i + size, j + size] = (1 / x1) * x2
    return filter


FILTER = create_filter(7)
FILTER_CENTER_INDEX = len(FILTER) // 2


image = Image.open('Pagani.jpg')
IMAGE_MATRIX = np.asarray(image, 'int32')

plt.imshow(IMAGE_MATRIX)
plt.show()

RED, GREEN, BLUE = IMAGE_MATRIX[:, :, 0], IMAGE_MATRIX[:, :, 1], IMAGE_MATRIX[:, :, 2]


for i in range(RED.shape[0] - (len(FILTER) - 1)):
    for j in range(RED.shape[1] - (len(FILTER) - 1)):

        RED[i + FILTER_CENTER_INDEX, j + FILTER_CENTER_INDEX] = np.sum(RED[i:i+len(FILTER), j:j+len(FILTER)] * FILTER / np.sum(FILTER))
        GREEN[i + FILTER_CENTER_INDEX, j + FILTER_CENTER_INDEX] = np.sum(GREEN[i:i+len(FILTER), j:j+len(FILTER)] * FILTER / np.sum(FILTER))
        BLUE[i + FILTER_CENTER_INDEX, j + FILTER_CENTER_INDEX] = np.sum(BLUE[i:i+len(FILTER), j:j+len(FILTER)] * FILTER / np.sum(FILTER))

blured_image = np.dstack((RED, GREEN, BLUE))



plt.imshow(blured_image)
plt.show()


