import numpy as np
from matplotlib import pyplot as plt
from skimage import color

import cv2

from utils.calculus import compute_hour_from_quarter, compute_minutes_from_quarter
from utils.custom_hough_transform import compute_hough_space, compute_maximums_vector
from utils.filters import otsu_threshold, get_image_contours
from utils.quarter_class import Quarter


def draw_lines(img, lines):
    draw_h, draw_w = img.shape
    draw_img = img.copy().astype(np.uint32)

    line_count = 2

    for line in lines:
        # r = 87
        r = line[0]
        # theta = 75 * np.pi / 180
        theta = line[1] * np.pi / 180
        for x in range(draw_w):
            try:
                y = int((r - x * np.cos(theta)) / np.sin(theta))

                if 0 < y < draw_h:
                    draw_img[x, y] = line_count
            except:
                pass

        line_count += 1

    plt.figure()
    plt.title("Draw image")
    plt.imshow(draw_img)
    plt.colorbar()


img = plt.imread('im/c2.png')

img = color.rgb2gray(img[:, :, :3]) * 255
plt.figure()
plt.imshow(img, cmap="gray")

h, w = img.shape
img = img[h // 3:2 * h // 3, w // 3: 2 * w // 3]
plt.figure()
plt.title("cropped")
plt.imshow(img, cmap="gray")

h, _ = np.histogram(img, bins=256, range=(0, 256), density=False)

plt.figure()
plt.plot(h)

print(otsu_threshold(h))

contoured_img = get_image_contours(img)
contour_histo, _ = np.histogram(contoured_img, bins=256, range=(0, 256), density=False)
threshold = otsu_threshold(contour_histo)
contoured_img = contoured_img > threshold

h, w = contoured_img.shape
contoured_img = contoured_img[5:h - 5 + 1, 5:w - 5 + 1]
h, w = contoured_img.shape

# hourly order not trigonometric
# quarter one starts at 12 and ends at 3
quarter_1 = contoured_img[: h // 2 + 1, w // 2:]
quarter_2 = contoured_img[h // 2:, w // 2:]
quarter_3 = contoured_img[h // 2:, : w // 2 + 1]
quarter_4 = contoured_img[: h // 2 + 1, : w // 2 + 1]

plt.figure()
plt.title("quarter1")
plt.imshow(quarter_1, cmap="gray")

plt.figure()
plt.title("quarter2")
plt.imshow(quarter_2, cmap="gray")

plt.figure()
plt.title("quarter3")
plt.imshow(quarter_3, cmap="gray")

plt.figure()
plt.title("quarter4")
plt.imshow(quarter_4, cmap="gray")

h, w = contoured_img.shape
contoured_img = contoured_img[5:h - 5 + 1, 5:w - 5 + 1]
plt.figure()
plt.title("contoured image")
plt.imshow(contoured_img, cmap='gray')

quarter_1 = Quarter(1, quarter_1)
quarter_2 = Quarter(2, quarter_2)
quarter_3 = Quarter(3, quarter_3)
quarter_4 = Quarter(4, quarter_4)
quarters = [quarter_1, quarter_2, quarter_3, quarter_4]
# sort quarters by number of pixels
# my assumption is that the minute line
# and the hour line will have the most pixels
# as usually those are the thicker ones
quarters.sort(key=lambda quarter: np.sum(quarter.data), reverse=True)
# pick the first 2 with the maximum number of pixels
quarters = quarters[:2]

# compute the hough space for each quarter
for quarter in quarters:
    hough_space = compute_hough_space(quarter.data)

    plt.figure()
    plt.title(f"hough space orig quarter {quarter.quarter}")
    plt.imshow(hough_space)
    plt.colorbar()

    # get rid of small accumulator numbers
    # which should not represent our lines
    # lines are found in the accumulators biggest numbers
    hough_space_threshold = np.max(hough_space) / 2
    hh, hw = hough_space.shape
    for line in range(hh):
        for column in range(hw):
            if hough_space[line, column] < hough_space_threshold:
                hough_space[line, column] = 0

    # to increase the peaks and decrease the lows in
    # the hough space, we will dilate it
    hough_space = cv2.dilate(hough_space, np.ones((3, 3)))

    # remember the hough_space in the objects variable
    quarter.hough_space = hough_space

    plt.figure()
    plt.title(f"hough space thresholded and dilated quarter {quarter.quarter}")
    plt.imshow(hough_space)
    plt.colorbar()

    # first element is the distance r
    # second element is the angle theta
    # third element represents how many pixels fit the line defined by r and theta

    lines = compute_maximums_vector(hough_space, maximums_to_return=5,
                                    value_sensitivity=15)

    print(lines)
    quarter.lines = lines

# usually the minute line is longer
# and sometimes thicker, so the most pixels
# will be the first in quarters list
hour = compute_hour_from_quarter(quarters[1])
minutes = compute_minutes_from_quarter(quarters[0])
print(f"Most probable time: {hour}:{minutes}")
hour = compute_hour_from_quarter(quarters[0])
minutes = compute_minutes_from_quarter(quarters[1])
print(f"Least probable time: {hour}:{minutes}")

plt.show()
