# Calculate histogram of frame
import cv2
import imutils
from imutils import paths
import numpy as np
import os


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


def trouver_images_bruit():
    # Define k mean clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS
    hist_frames = []

    list = os.listdir('./frames')  # dir is your directory path
    number_images = len(list)

    for i in range(0, number_images):
        # Histogram of each frame as feature for clustering
        frame = cv2.imread('./frames/frame%d.jpg' % i)
        hist = extract_color_histogram(frame)
        hist_frames.append(hist)

    hist_frames = np.float32(hist_frames)
    # cluster
    # Apply KMeans
    Z = hist_frames
    compactness, labels, centers = cv2.kmeans(Z, 2,
                                              criteria, 10, flags)

    count_un = 0

    for i in range(0, len(labels)):
        if labels[i] == 1:
            count_un += 1
    count_zero = len(labels) - count_un

    if count_un > count_zero:
        tab_bruit = [i for i, x in enumerate(labels) if x == 0]
    else:
        tab_bruit = [i for i, x in enumerate(labels) if x == 1]

    return tab_bruit
