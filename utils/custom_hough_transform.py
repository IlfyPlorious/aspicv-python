import matplotlib.pyplot as plt
import numpy as np
from skimage import color


def setup_accumulator(img, degrees_accuracy):
    img_h, img_w = img.shape
    h = int(np.ceil(np.sqrt(img_h ** 2 + img_w ** 2)))
    w = 180 // degrees_accuracy

    return np.zeros((h, w))


def compute_hough_space(img, degrees_accuracy):
    """
     ideally the image is black and white
     not grayscale, where lines are defined with 1 (255)
     and background is 0 (0)

     this method returns the hough space for this b/w image
     to get the lines you need to extract the r and theta for
     the maximum points in the hough transform space.
    """

    accumulator = setup_accumulator(img, degrees_accuracy)
    print(accumulator.shape)

    theta_angles = np.linspace(0, np.pi, 180)

    h, w = img.shape

    for line in range(0, h):
        for column in range(0, w):
            if img[line, column] > 0:
                for theta in theta_angles:
                    r = int(line * np.cos(theta) + column * np.sin(theta))
                    t = int(theta * 180 / np.pi) - 1
                    accumulator[r, t] += 1

    return accumulator


img = plt.imread('../im/b_w_test.png')
img = color.rgb2gray(img)

h, w = img.shape
for line in range(0, h):
    for column in range(0, w):
        if img[line, column] > 0.5:
            img[line, column] = 0
        else:
            img[line, column] = 1

plt.figure()
plt.imshow(img, cmap='gray')
plt.show()

hough_space = compute_hough_space(img, 1)
print(hough_space.shape)
print(hough_space)

plt.figure()
plt.imshow(hough_space)
plt.show()
