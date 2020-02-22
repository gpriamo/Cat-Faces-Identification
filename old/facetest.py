import time

import cv2.cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

models_dir = '../models/detection/'
# try: SF= 1.1 - N = 5
SF = 1.02  # 1.015  # play around with it (i.e. 1.05, 1.3 etc) Good ones: 1.04 (haar), 1.05
N = 2  # play around with it (3,4,5,6) Good ones: 2 (haar)


def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
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


def cat_face_detect(file, det=0):
    im_orig = cv.imread(file)

    # img = image_resize(im_orig, 512, 512)

    # recognizer = cv.face.LBPHFaceRecognizer_create()
    # print(recognizer.train())

    img = im_orig

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cat_cascade = cv.CascadeClassifier(models_dir + 'haarcascade_frontalcatface.xml')
    cat_cascade_ext = cv.CascadeClassifier(models_dir + 'haarcascade_frontalcatface_extended.xml')
    cat_cascade_lbp = cv.CascadeClassifier(models_dir + 'lbpcascade_frontalcatface.xml')
    eye_cascade = cv.CascadeClassifier(models_dir + 'haarcascade_eye.xml')

    print(cat_cascade.empty())
    print(cat_cascade_ext.empty())
    print(cat_cascade_lbp.empty())

    if det == 0:
        print("Normal Haar:")
        t0 = cv.getTickCount()
        cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)
        t1 = cv.getTickCount()
        compute_elapsed_time(t0, t1)
        time.sleep(2)

    elif det == 1:
        print("Extended Haar:")
        t0 = cv.getTickCount()
        cats_ext = cat_cascade_ext.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)
        t1 = cv.getTickCount()
        compute_elapsed_time(t0, t1)
        time.sleep(2)

    elif det == 2:
        print("LBP:")
        t0 = cv.getTickCount()
        cats_lbp = cat_cascade_lbp.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)
        t1 = cv.getTickCount()
        compute_elapsed_time(t0, t1)

    if det == 0:
        for (x, y, w, h) in cats:  # blue = haar
            img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.08, minNeighbors=3)
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.08, minNeighbors=3, minSize=(40, 40))
            print(len(eyes))
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    elif det == 1:
        for (x, y, w, h) in cats_ext:  # green = haar ext
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            # eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5)
            print(len(eyes))
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    elif det == 2:
        for (x, y, w, h) in cats_lbp:  # red = LBP
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5)
            print(len(eyes))
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    cv.namedWindow('win', cv.WINDOW_NORMAL)
    # cv.resizeWindow('win', 1980, 1800)

    cv.imshow('win', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def compute_elapsed_time(t0, t1):
    secs = (t1 - t0) / cv.getTickFrequency()
    print("Elapsed time: {} seconds".format(secs))


def face_detect():
    img = cv.imread('../images/random/rami.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(models_dir + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier(models_dir + 'haarcascade_eye.xml')

    print(face_cascade.empty())

    # cv.namedWindow("camera",1)
    # capture = cv.VideoCapture()
    # capture.open(0)
    # while True:
    #     img = capture.read()[1]
    #     cv.imshow("camera", img)
    #     if cv.waitKey(10) == 27: break
    # cv.destroyWindow("camera")

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(str(type(faces)) + " " + str(type(faces[0])))
    for (x, y, w, h) in faces:
        print(x)
        print(type(x))
        print(y)
        print(type(y))
        print(w)
        print(type(w))
        print(type(h))
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print(len(eyes))
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    imdir = '../images/random/'
    imdir = '../images/dataset/unprocessed/c/'

    images = [os.path.join(imdir, f) for f in os.listdir(imdir) if
              os.path.isfile(os.path.join(imdir, f))]

    print(images)

    for im in images:
        if im.split('/')[-1][0:-5] not in ['c10']:
            continue
        print("Working on " + str(im.split('/')[-1][0:-4]))
        print(im)
        cat_face_detect(im, 0)
        cat_face_detect(im, 1)
        cat_face_detect(im, 2)
