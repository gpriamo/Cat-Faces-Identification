import cv2.cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

models_dir = '../models/'
SF = 1.05  # play around with it (i.e. 1.3 etc)
N = 3   # play around with it (3,4,5,6)


def cat_face_detect(im):
    img = cv.imread(im)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cat_cascade = cv.CascadeClassifier(models_dir + 'haarcascade_frontalcatface.xml')
    cat_cascade_ext = cv.CascadeClassifier(models_dir + 'haarcascade_frontalcatface_extended.xml')

    print(cat_cascade.empty())
    print(cat_cascade_ext.empty())

    cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)
    cats_ext = cat_cascade_ext.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)

    for (x, y, w, h) in cats:
        # blue
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in cats_ext:
        # green
        img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.namedWindow('win', cv.WINDOW_NORMAL)
    #cv.resizeWindow('win', )

    cv.imshow('win', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def face_detect():
    img = cv.imread('../images/rami.jpg')
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
    #face_detect()
    im1 = '../images/Cat03.jpg'
    im2 = '../images/t1.jpg'
    im3 = '../images/cat2.jpg'
    im4 = '../images/cat4.jpeg' # bad res


    im = im1

    cat_face_detect(im)
