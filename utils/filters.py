import numpy as np
from matplotlib import pyplot as plt


def otsu_threshold(h):
    eps = 0.0000000000001
    criteria = np.zeros(256)
    L = 256
    for T in range(0, L):
        P0 = 0
        mu0 = 0
        for i in range(0, T):
            P0 += h[i]
            mu0 += i * h[i]
        mu0 = mu0 / (P0 + eps)

        P1 = 0
        mu1 = 0
        for i in range(T, L):
            P1 += h[i]
            mu1 += i * h[i]
        mu1 = mu1 / (P1 + eps)
        criteria[T] = P0 * mu0 * mu0 + P1 * mu1 * mu1

    thr = np.argmax(criteria)
    return thr


def get_median_value(array: list):
    array_copy = array.copy()
    array_copy.sort()
    return array[len(array) // 2]


def apply_median_filter(img, kernel_size=5):
    h, w = img.shape
    half_size = kernel_size // 2
    for line in range(half_size, h - half_size):
        for column in range(half_size, w - half_size):
            kernel = img[line - half_size: line + half_size + 1, column - half_size: column + half_size + 1]
            median = get_median_value(kernel.flatten())
            img[line, column] = median


def get_image_contours(img=np.zeros((1, 1)), show_plots=False):
    h, w = img.shape

    bordered_image = np.zeros((h + 1, w + 1))
    bordered_image[:h, :w] = img

    fx = np.zeros((h, w))
    fy = np.zeros((h, w))

    for line in range(0, h):
        for column in range(0, w):
            fx[line, column] = bordered_image[line, column] - bordered_image[line + 1, column]
            fy[line, column] = bordered_image[line, column] - bordered_image[line, column + 1]

    if show_plots:
        plt.figure(), plt.imshow(fx, cmap='gray'), plt.colorbar(), plt.show()
        plt.figure(), plt.imshow(fy, cmap='gray'), plt.colorbar(), plt.show()

    return np.abs(fx) + np.abs(fy)
