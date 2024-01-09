import numpy as np


def setup_accumulator(img, degrees_accuracy):
    img_h, img_w = img.shape
    h = int(np.ceil(np.sqrt(img_h ** 2 + img_w ** 2)))
    w = 180

    return np.zeros((h, w))


def compute_window_for_angle(real_angle, window_size=3):
    window = np.zeros((window_size, window_size), dtype=np.float64)
    center = (window_size - 1) // 2
    for line in range(window_size):
        for column in range(window_size):
            if line - center == 0:
                point_angle = 0
            elif column - center == 0:
                point_angle = np.pi / 2
            else:
                point_angle = np.pi - np.arctan((column - center) / (line - center))
            if np.abs(point_angle - real_angle) < 0.00001:
                abs_weight = 0.00001
            else:
                abs_weight = np.abs(point_angle - real_angle)

            window[line, column] = 1 / abs_weight

    window[center, center] = np.max(window)

    return window / np.sum(window)


def compute_hough_space(img, degrees_accuracy=1):
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

    window_size = 3
    window_threshold = 0.4
    radius = (window_size - 1) // 2
    img = img.astype(np.int32)
    for line in range(radius, h - radius):
        for column in range(radius, w - radius):
            if img[line, column] > 0:
                for theta in theta_angles:
                    # real_angle = theta + np.pi / 2
                    # weight_window = compute_window_for_angle(real_angle, window_size)
                    # window = img[line - radius: line + radius + 1,
                    #          column - radius: column + radius + 1] * weight_window
                    # window_density = np.sum(window)
                    # if window_density > window_threshold:
                    #     r = int(line * np.cos(theta) + column * np.sin(theta))
                    #     t = int(theta * 180 / np.pi) - 1
                    #     accumulator[r, t] += 1
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
    :return: maximums vector where index 0 is r, index 1 is theta and index 2 is the number of points on that line;
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
        should_skip = False
        for vector in vector_to_return:
            if abs(sorted_maximums[index][1] - vector[1]) < value_sensitivity:
                should_skip = True

        if should_skip:
            index += 1
            continue
        else:
            vector_to_return.append(sorted_maximums[index])
            index += 1

    return vector_to_return
