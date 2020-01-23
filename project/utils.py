import cv2.cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

SF = 1.05  # play around with it (i.e. 1.05, 1.3 etc) Good ones: 1.04 (haar), 1.05
N = 2  # play around with it (3,4,5,6) Good ones: 2 (haar)
cascade_models_dir = '../models/detection/'
cat_cascades = ['haarcascade_frontalcatface.xml', 'haarcascade_frontalcatface_extended.xml',
                'lbpcascade_frontalcatface.xml']


def resize_image(image, width=None, height=None, inter=cv.INTER_AREA):
    """
    Resizes an image according to the specified parameters.

    :param image: image
        image to resize
    :param width: int
        output width
    :param height: int
        output height
    :param inter: interpolation
        interpolation to use
    :return: resized image
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def detect_cat_face(file, classifier, show=False, scaleFactor=SF, minNeighbors=N):
    """
    Cat face detection utility.

    :param file : str
        The name of the image file to detect the face from.
    :param classifier : int
        Integer used to select the type of detector model to be used:
        0 = haarcascade_frontalcatface.xml
        1 = haarcascade_frontalcatface_extended.xml
        2 = lbpcascade_frontalcatface.xml
    :param show: set to True to see an output image
    :param scaleFactor : float
        Scale factor value the detector should use
    :param minNeighbors : int
        Min neighbors value the detector should use
    :return a list of rectangles containing the detected features
    """

    cat_cascade = cv.CascadeClassifier(cascade_models_dir + cat_cascades[classifier])

    if cat_cascade.empty():
        raise RuntimeError('The classifier was not loaded correctly!')

    img = cv.imread(file)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)

    if show:
        for (x, y, w, h) in face:  # blue
            img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.namedWindow('win', cv.WINDOW_NORMAL)
        # cv.resizeWindow('win', 1980, 1800)

        cv.imshow('win', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return face


if __name__ == '__main__':
    """Main for testing purposes"""
    imdir = ''

    images = [os.path.join(imdir, f) for f in os.listdir(imdir) if
              os.path.isfile(os.path.join(imdir, f))]

    for im in images:
        detect_cat_face(im, 0, show=True)
