# import the necessary packages
import argparse
import numpy as np
import cv2
import heapq
import os
from operator import itemgetter
import sample


# Script permet de decouper le video en images
def decoupe_video(video):
    print'--Decomposition de video ...'
    # vérifier l'existance de dossier frames dans lequel on va stocker les images, s'il n'existe pas on va le créer,
    # sinon il faut le vider pour qu'il soit pret
    if not os.path.exists('frames'):
        os.makedirs('frames')
    else:
        files = os.listdir('frames')
        for i in range(0, len(files)):
            os.remove('frames' + '/' + files[i])

    try:
        # capture de video
        cap = cv2.VideoCapture(video)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print '_Extraire frame%d' % count
                cv2.imwrite("./frames/frame%d.jpg" % count, frame)
                count = count + 1
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cap.destroyAllWindows()
    except:
        pass


# on calcule l'erreur mse (Mean Square Error) entre 2 images
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def compare_images(imageA, imageB):
    image1 = cv2.imread(imageA)
    image2 = cv2.imread(imageB)
    # on transforme les images en COLOR_BGR2GRAY pour facilité le calcule et pour qu'il soit plus rapide
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # calculer l'erreur entre les 2 images :
    m = mse(image1_gray, image2_gray)
    return m


# suppression de bruit en utilisant machine learning,
def supprimer_bruit(tab):
    # recuperer les images etranges dans la liste des images
    tab_bruit = sample.trouver_images_bruit()
    s = set(tab_bruit)
    # enlever les images de la liste
    tab_sans_bruit = [x for x in tab if x not in s]
    # retourner les images sans bruit
    return tab_sans_bruit


# Créer la matrice qui contiendra la distance entre chaque images et les autres
def matrix_distance(number_images):
    matrix1 = [[0] * number_images for i in range(number_images)]
    print '--Reorganiser les images ...'
    for i in range(0, number_images):
        for j in range(0, number_images):
            # calculer l'erreur entre l'image framei.jpg et l'image framej.jpg
            m = compare_images("./frames/frame%d.jpg" % i, "./frames/frame%d.jpg" % j)
            # enregister l'erreur dans la matrice
            matrix1[i][j] = m
        print '--Recherche de l`image suivante de l`image : %d' % i
    return matrix1


# trier les images en se basant sur la matrice
def trier_images(matrix, number_images):
    tab_trier = []
    tab_trier.append(0)
    # trier les images dans la table tab_trier
    for i in range(1, number_images):
        k = 1
        ajout = False
        # trouver l'image la l'erreur le plus faible avec l'image i, et on recupere son indice
        ind, val = heapq.nsmallest(k, enumerate(matrix[tab_trier[i - 1]]), itemgetter(1))[-1]
        while ajout == False:
            # si l'image n'existe pas déja dans le tableau on l'ajoute
            if ind not in tab_trier and ind != i:
                tab_trier.append(ind)
                ajout = True
            else:
                # si l'image existe déja dans le tableau on cherche l'image suivante avec l'erreur le plus faible
                k += 1
                ind, val = heapq.nsmallest(k, enumerate(matrix[tab_trier[i - 1]]), itemgetter(1))[-1]
    return tab_trier


# Chercher le premier frame dans le video pour reorganiser les scenes
def trouver_premier_frame(tab_sans_bruit):
    tab_dist = []
    tab_dist.append(0)
    # il faut chercher l'erreur le plus grand entre trois images pour reorganiser les scenes
    for i in range(1, len(tab_sans_bruit) - 1):
        dist1 = compare_images("./frames/frame%d.jpg" % tab_sans_bruit[i - 1],
                               "./frames/frame%d.jpg" % tab_sans_bruit[i])
        dist2 = compare_images("./frames/frame%d.jpg" % tab_sans_bruit[i],
                               "./frames/frame%d.jpg" % tab_sans_bruit[i + 1])
        tab_dist.append(abs(dist1 - dist2))
    # recuperer l'index de l'image autour de laquelle il y'a l'erreur le plus grand
    ind, val = heapq.nlargest(1, enumerate(tab_dist), itemgetter(1))[-1]
    # déplacer les images qui se trouve apres l'index pour organiser le video
    tab_org = tab_sans_bruit[ind + 1:len(tab_sans_bruit)] + tab_sans_bruit[0:ind + 1]

    return tab_org


# recreation de video
def recreer_video(tab_video):
    print '--Suppression de bruit ...'
    # recuperer la table des images en supprimant les images etranges
    tab_sans_bruit = supprimer_bruit(tab_video)
    print '--Recherche de premier frame ...'
    tab_final = trouver_premier_frame(tab_sans_bruit)
    from cv2 import VideoWriter, imread, resize
    tab_final = tab_final[::-1]
    size = None
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    vid = None
    # recréer le video corriger avec le nom video_corrige.avi
    for i in range(0, len(tab_final) - 1):
        k = tab_final[i]
        img = imread('./frames/frame%d.jpg' % k)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter('./video_corrige.avi', fourcc, float(25), size, True)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    print '*********** Video cree ***************'
    vid.release()


def main():
    # les arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
                    help="path to input video")
    args = vars(ap.parse_args())

    # recuperer le nom de video
    videoPaths = args["video"]
    # decouper le video en images
    decoupe_video(videoPaths)
    list = os.listdir('./frames')  # dir is your directory path
    # calculer le nombre d'images dans le dossier frames
    number_images = len(list)
    # la matrice qui la distance entre chaque image et les autres
    matrix1 = matrix_distance(number_images)
    # trier les images dans le tableau tab_video
    tab_video = trier_images(matrix1, number_images)
    # lancer la recreation de video
    recreer_video(tab_video)


if __name__ == '__main__':
    main()
