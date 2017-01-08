# import the necessary packages
import argparse

import numpy as np
import cv2
import heapq
import os
from operator import itemgetter

import sample


def decoupe_video(video):
    print'--Decomposition de video ...'
    if not os.path.exists('frames'):
        os.makedirs('frames')
    else:
        files = os.listdir('frames')
        for i in range(0, len(files)):
            os.remove('frames' + '/' + files[i])

    try:
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


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def compare_images(imageA, imageB):
    image1 = cv2.imread(imageA)
    image2 = cv2.imread(imageB)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    m = mse(image1_gray, image2_gray)
    return m


def supprimer_bruit(tab):
    tab_bruit = sample.trouver_images_bruit()
    s = set(tab_bruit)
    tab_sans_bruit = [x for x in tab if x not in s]
    return tab_sans_bruit


def trouver_premier_frame(tab_sans_bruit):
    tab_dist = []
    tab_dist.append(0)
    for i in range(1, len(tab_sans_bruit) - 1):
        dist1 = compare_images("./frames/frame%d.jpg" % tab_sans_bruit[i - 1], "./frames/frame%d.jpg" % tab_sans_bruit[i])
        dist2 = compare_images("./frames/frame%d.jpg" % tab_sans_bruit[i], "./frames/frame%d.jpg" % tab_sans_bruit[i + 1])
        tab_dist.append(abs(dist1 - dist2))
    ind, val = heapq.nlargest(1, enumerate(tab_dist), itemgetter(1))[-1]
    tab_org = tab_sans_bruit[ind + 1:len(tab_sans_bruit)] + tab_sans_bruit[0:ind + 1]

    return tab_org


def recreer_video(tab_video):
    print '--Suppression de bruit ...'
    tab_sans_bruit = supprimer_bruit(tab_video)
    print '--Recherche de premier frame ...'
    tab_final = trouver_premier_frame(tab_sans_bruit)
    from cv2 import VideoWriter, imread, resize
    tab_final = tab_final[::-1]
    size = None
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    vid = None
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
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
                    help="path to input video")
    args = vars(ap.parse_args())

    videoPaths = args["video"]

    decoupe_video(videoPaths)
    list = os.listdir('./frames')  # dir is your directory path
    number_images = len(list)

    matrix1 = [[0] * number_images for i in range(number_images)]
    tab_video = []
    print '--Reorganiser les images ...'
    for i in range(0, number_images):
        for j in range(0, number_images):
            m = compare_images("./frames/frame%d.jpg" % i, "./frames/frame%d.jpg" % j)
            matrix1[i][j] = m
        print '--Recherche de l`image suivante de l`image : %d' % i
    tab_video.append(0)
    for i in range(1, number_images):
        k = 1
        ajout = False
        ind, val = heapq.nsmallest(k, enumerate(matrix1[tab_video[i - 1]]), itemgetter(1))[-1]
        while ajout == False:
            if ind not in tab_video and ind != i:
                tab_video.append(ind)
                ajout = True
            else:
                k += 1
                ind, val = heapq.nsmallest(k, enumerate(matrix1[tab_video[i - 1]]), itemgetter(1))[-1]

    recreer_video(tab_video)


if __name__ == '__main__':
    main()
