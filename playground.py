import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import color

from utils.custom_hough_transform import get_masked_median, compute_hough_space, compute_maximums_vector
from utils.filters import otsu_threshold, apply_median_filter, get_image_contours

img = plt.imread('im/c1.png')

img = color.rgb2gray(img[:, :, :3]) * 255
plt.figure()
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

hough_space = compute_hough_space(contoured_img, area=1)
plt.figure()
plt.imshow(hough_space)
# first element is the distance r
# second element is the angle theta
# third element represents how many pixels fit the line defined by r and theta
lines = compute_maximums_vector(hough_space, avg_mask=np.array([[1]]), maximums_to_return=6)

print(lines)

draw_h, draw_w = contoured_img.shape
draw_img = contoured_img.copy()

for line in lines:
    # r = 87
    r = line[0]
    # theta = 75 * np.pi / 180
    theta = line[1] * np.pi / 180
    for x in range(draw_w):
        try:
            y = int((r - x * np.cos(theta)) / np.sin(theta))

            if 0 < y < draw_h:
                draw_img[x, y] = 5
        except:
            pass

plt.figure()
plt.imshow(draw_img)

plt.show()