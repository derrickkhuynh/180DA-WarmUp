#code taken from example at 

#https://github.com/opencv/opencv/blob/master/samples/python/kmeans.py
#https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097

import sys
import video

import cv2
import numpy as np
import matplotlib.pyplot as plt
from gaussian_mix import make_gaussians
from sklearn.cluster import KMeans

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    #modified so it only plots the first color (the largest) instead of all colors
    for (percent, color) in zip(hist, centroids):
        cv2.rectangle(bar, (0, 0), (300, 50),
                      color.astype("uint8").tolist(), -1)
        break

    return bar

def main():
    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv2.namedWindow('Video Capture')
    cv2.namedWindow('Dominant Color in center 200x200px square')

    cap = video.create_capture(fn)
    while True:
        _flag, img = cap.read()
        crop_img = img[200:400, 200:400]
        edit_img = crop_img.reshape((crop_img.shape[0] * crop_img.shape[1],3)) #represent as row*column,channel number
        clt = KMeans(n_clusters=3) #cluster number
        clt.fit(edit_img)

        hist = find_histogram(clt)
        bar = plot_colors2(hist, clt.cluster_centers_)

        cv2.rectangle(img, (200, 200), (400,400), (0, 255 , 0))

        cv2.imshow('Dominant Color in center 200x200px square', bar)
        cv2.imshow('Video Capture', img)
        ch = cv2.waitKey(5)
        if ch == 27:
            break

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()