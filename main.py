from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from os import walk

SIGMA = 2


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


def blur(image_matrix, kernel):
    '''
    ATTENTION!!!
    for loops!!!
    SLOW PROCESSING
    :param image_matrix:
    :param kernel:
    :return blured image :
    '''
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
    # for i in range(red.shape[0] - (len(FILTER) - 1)):
    #     for j in range(red.shape[1] - (len(FILTER) - 1)):
    #
    #         red[i + FILTER_CENTER_INDEX, j + FILTER_CENTER_INDEX] = np.sum(red[i:i+len(kernel), j:j+len(kernel)] *
    #                                                                        kernel / np.sum(kernel))
    #         green[i + FILTER_CENTER_INDEX, j + FILTER_CENTER_INDEX] = np.sum(green[i:i+len(kernel), j:j+len(kernel)] *
    #                                                                          kernel / np.sum(kernel))
    #         blue[i + FILTER_CENTER_INDEX, j + FILTER_CENTER_INDEX] = np.sum(blue[i:i+len(kernel), j:j+len(kernel)] *
    #                                                                         kernel / np.sum(kernel))

    temp_array = np.zeros((red.shape[0] - (len(FILTER) - 1), red.shape[1] - (len(FILTER) - 1)))
    it = np.nditer(temp_array, flags=['multi_index'])
    for _ in it:
        red[it.multi_index[0] + FILTER_CENTER_INDEX, it.multi_index[1] + FILTER_CENTER_INDEX] = np.sum(red[it.multi_index[0]:it.multi_index[0] + len(kernel), it.multi_index[1]:it.multi_index[1] + len(kernel)] *
                                                                      kernel / np.sum(kernel))
        green[it.multi_index[0] + FILTER_CENTER_INDEX, it.multi_index[1] + FILTER_CENTER_INDEX] = np.sum(green[it.multi_index[0]:it.multi_index[0]+len(kernel), it.multi_index[1]:it.multi_index[1] + len(kernel)] *
                                                                        kernel / np.sum(kernel))
        blue[it.multi_index[0] + FILTER_CENTER_INDEX, it.multi_index[1] + FILTER_CENTER_INDEX] = np.sum(blue[it.multi_index[0]:it.multi_index[0]+len(kernel), it.multi_index[1]:it.multi_index[1] + len(kernel)] *
                                                                       kernel / np.sum(kernel))

    # crop
    red = red[len(kernel) // 2:red.shape[0] - len(kernel) // 2, len(kernel) // 2:red.shape[1] - len(kernel) // 2]
    green = green[len(kernel) // 2:green.shape[0] - len(kernel) // 2, len(kernel) // 2:green.shape[1] - len(kernel) // 2]
    blue = blue[len(kernel) // 2:blue.shape[0] - len(kernel) // 2, len(kernel) // 2:blue.shape[1] - len(kernel) // 2]

    print(red.shape, green.shape, blue.shape)

    image = np.dstack((red, green, blue))
    return image


def rgb_convolve(image, kern, avg_brightness_default):
    image2 = np.empty_like(image)
    for dim in range(image.shape[-1]):
        image2[:, :, dim] = sp.signal.convolve2d(
            image[:, :, dim],
            kern,
            mode='same',
            boundary='symm'
        )

    r, g, b = image2[:, :, 0], image2[:, :, 1], image2[:, :, 2]
    avg_r, avg_g, avg_b = np.average(r), np.average(g), np.average(b)
    avg_brightness = np.average([avg_r, avg_g, avg_g])
    print(avg_brightness)
    difference = avg_brightness_default - avg_brightness
    image2[:, :, 0] += int(difference)
    image2[:, :, 1] += int(difference)
    image2[:, :, 2] += int(difference)

    return image2


def load_images(path):
    images = []
    for _, __, img_files in walk(path):
        for image in img_files:
            images.append(np.asarray(Image.open(path + '/' + image), 'int32'))
    return images


def load_video(path):
    import cv2
    capture = cv2.VideoCapture(path)
    print(capture)

load_video('videos')


def blur_multiple_images(images):
    for image in images:
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        avg_r, avg_g, avg_b = np.average(r), np.average(g), np.average(b)
        avg_brightness_default = np.average([avg_r, avg_g, avg_g])

        plt.imshow(rgb_convolve(image, FILTER, avg_brightness_default))
        plt.show()

# image = Image.open('test.jpg')
# IMAGE = np.asarray(image, 'int32')
#
# r, g, b = IMAGE[:, :, 0], IMAGE[:, :, 1], IMAGE[:, :, 2]
#
# avg_r, avg_g, avg_b = np.average(r), np.average(g), np.average(b)
#
# avg_brightness_default = np.average([avg_r, avg_g, avg_g])
#
#
# plt.imshow(IMAGE)
# plt.show()
#
#
#
#
# plt.imshow(rgb_convolve(IMAGE, FILTER))
# plt.show()


