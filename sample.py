# Calculate histogram of frame
import cv2
import imutils
import numpy as np
import os


def extract_color_histogram(image, bins=(8, 8, 8)):
    # Extraire un histogramme de couleur 3D de l'espace de couleur HSV en
    # utilisant le nombre de «bins» fournis par canal
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # Gérer la normalisation de l'histogramme si nous utilisons OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # Sinon, effectuer une normalisation "en place" dans OpenCV 3
    else:
        cv2.normalize(hist, hist)

    # Renvoie l'histogramme comme vecteur caractéristique
    return hist.flatten()


def trouver_images_bruit():
    # Define k mean clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Juste pour éviter la coupure de ligne dans le code)
    flags = cv2.KMEANS_RANDOM_CENTERS
    hist_frames = []

    list = os.listdir('./frames')  # dir is your directory path
    number_images = len(list)

    for i in range(0, number_images):
        # Histogramme de chaque frame en tant que caractéristique pour le clustering
        frame = cv2.imread('./frames/frame%d.jpg' % i)
        hist = extract_color_histogram(frame)
        # construit la liste d'histogrammes
        hist_frames.append(hist)
    # transformation de la liste en np.float qui acceptable par k mean clustring
    hist_frames = np.float32(hist_frames)
    # cluster
    Z = hist_frames
    compactness, labels, centers = cv2.kmeans(Z, 2,
                                              criteria, 10, flags)

    # le k cluster va trier les features dans 2 catégorie (labels)
    # on va calculer le nombre de images identifier par '1' et le nombre d'images identifier par '0'
    count_un = 0
    for i in range(0, len(labels)):
        if labels[i] == 1:
            count_un += 1
    count_zero = len(labels) - count_un

    # on elimine la minorité qui represente les images bruit
    if count_un > count_zero:
        tab_bruit = [i for i, x in enumerate(labels) if x == 0]
    else:
        tab_bruit = [i for i, x in enumerate(labels) if x == 1]

    return tab_bruit
