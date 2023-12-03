import matplotlib.pyplot as plt
import numpy as np
from skimage import color


def setup_accumulator(img, degrees_accuracy):
    img_h, img_w = img.shape
    h = int(np.ceil(np.sqrt(img_h ** 2 + img_w ** 2)))
    w = 180 // degrees_accuracy

    return np.zeros((h, w))


def compute_hough_space(img, area=3, degrees_accuracy=1):
    """
     ideally the image is black and white
     not grayscale, where lines are defined with 1 (255)
     and background is 0 (0)

     this method returns the hough space for this b/w image
     to get the lines you need to extract the r and theta for
     the maximum points in the hough transform space.

     everything that is < 90 has positive slope
     everything that is > 90 has negative slope
    """

    accumulator = setup_accumulator(img, degrees_accuracy)

    theta_angles = np.linspace(0, np.pi, 180 // degrees_accuracy)

    h, w = img.shape

    for line in range(0, h, area):
        for column in range(0, w, area):
            if img[line, column] > 0:
                for theta in theta_angles:
                    r = int(line * np.cos(theta) + column * np.sin(theta))
                    t = int(theta * 180 / np.pi) - 1
                    accumulator[r, t] += 1

    return accumulator


def get_masked_median(matrix_area, mask=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])):
    lines, columns = mask.shape
    sum = 0
    for line in range(0, lines):
        for column in range(0, columns):
            sum += matrix_area[line, column] * mask[line, column]

    return sum / (lines * columns)


def compute_maximums_vector(matrix,
                            avg_mask=np.array([
                                [1, 1, 1], [1, 1, 1], [1, 1, 1]
                            ]),
                            value_sensitivity=4,
                            maximums_to_return=2):
    """
    :param matrix: matrix to compute maximums for
    :param avg_mask: mask to avg points in the matrix to avoid clustered maximums in one point
    :param value_sensitivity: by how much the sorting criteria should vary
    :param maximums_to_return: how many maximums to return
    :return: maximums vector where index 0 is the y coordinate, index 1 is x coordinate and index 2 is the point value;
    the vector is sorted desc by value
    """

    mask_h, mask_w = avg_mask.shape
    lines, columns = matrix.shape

    maximums = []

    for line in range(mask_h // 2, lines - mask_h // 2, mask_h):
        for column in range(mask_w // 2, columns - mask_w // 2, mask_w):
            maximums.append(
                (line, column, matrix[line, column]))

    sorted_maximums = sorted(maximums, key=lambda maximum: maximum[2], reverse=True)
    index = 1

    vector_to_return = [sorted_maximums[0]]

    while len(vector_to_return) < maximums_to_return and index < len(sorted_maximums):
        if abs(sorted_maximums[index][1] - sorted_maximums[index - 1][1]) < value_sensitivity:
            index += 1
        else:
            vector_to_return.append(sorted_maximums[index])
            index += 1

    return vector_to_return
