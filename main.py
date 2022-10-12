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
IMAGE = np.asarray(image, 'int32')

plt.imshow(IMAGE)
plt.show()


def blur(image_matrix, kernel):
    red, green, blue = image_matrix[:, :, 0], image_matrix[:, :, 1], image_matrix[:, :, 2]
    print(red.shape, green.shape)

    # padding
    red = np.pad(red, ((len(kernel) // 2, len(kernel) // 2), (len(kernel) // 2, len(kernel) // 2)), mode='constant',
                 constant_values=0)
    green = np.pad(green, ((len(kernel) // 2, len(kernel) // 2), (len(kernel) // 2, len(kernel) // 2)), mode='constant',
                   constant_values=0)
    blue = np.pad(blue, ((len(kernel) // 2, len(kernel) // 2), (len(kernel) // 2, len(kernel) // 2)), mode='constant',
                  constant_values=0)
    print(red.shape, green.shape)

    # main loop
    for i in range(red.shape[0] - (len(FILTER) - 1)):
        for j in range(red.shape[1] - (len(FILTER) - 1)):

            red[i + FILTER_CENTER_INDEX, j + FILTER_CENTER_INDEX] = np.sum(red[i:i+len(kernel), j:j+len(kernel)] *
                                                                           kernel / np.sum(kernel))
            green[i + FILTER_CENTER_INDEX, j + FILTER_CENTER_INDEX] = np.sum(green[i:i+len(kernel), j:j+len(kernel)] *
                                                                             kernel / np.sum(kernel))
            blue[i + FILTER_CENTER_INDEX, j + FILTER_CENTER_INDEX] = np.sum(blue[i:i+len(kernel), j:j+len(kernel)] *
                                                                            kernel / np.sum(kernel))
    # crop
    red = red[len(kernel) // 2:red.shape[0] - len(kernel) // 2, len(kernel) // 2:red.shape[1] - len(kernel) // 2]
    green = green[len(kernel) // 2:green.shape[0] - len(kernel) // 2, len(kernel) // 2:green.shape[1] - len(kernel) // 2]
    blue = blue[len(kernel) // 2:blue.shape[0] - len(kernel) // 2, len(kernel) // 2:blue.shape[1] - len(kernel) // 2]

    print(red.shape, green.shape, blue.shape)

    image = np.dstack((red, green, blue))
    return image


plt.imshow(blur(IMAGE, FILTER))
plt.show()


