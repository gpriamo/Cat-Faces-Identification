import math
import cv2.cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image

# def_SF = 1.05  # play around with it (i.e. 1.05, 1.3 etc) Good ones: 1.04 (haar), 1.05
# def_N = 2  # play around with it (3,4,5,6) Good ones: 2 (haar)
cascade_models_dir = '../models/detection/'
cat_cascades = ['haarcascade_frontalcatface.xml', 'haarcascade_frontalcatface_extended.xml',
                'lbpcascade_frontalcatface.xml']
eye_cascade_model = cascade_models_dir + 'haarcascade_eye.xml'


def detect_cat_face(file, classifier, show=False, scaleFactor=1.05, minNeighbors=2,
                    eyes_ScaleFactor=1.08, eyes_minNeighbors=3, eyes_minSize=(40, 40)):
    """
    Cat face detection utility.

    :param file : str
        The name of the image file to detect the face from.
    :param classifier : int
        Integer used to select the type of detector model to be used:
        0 = haarcascade_frontalcatface.xml
        1 = haarcascade_frontalcatface_extended.xml
        2 = lbpcascade_frontalcatface.xml
    :param show: bool
        set to True to see an output image
    :param scaleFactor: float
        Scale factor value the detector should use
    :param minNeighbors : int
        Min neighbors value the detector should use
    :param eyes_ScaleFactor: float
        scaleFactor value the eyes detector should use
    :param eyes_minNeighbors:
        minNeighbors value the eyes detector should use
    :param eyes_minSize:
        minSize value the eyes detector should use
    :return the cropped face and the location of the eyes, if detected, else None.
    """

    detector = cat_cascades[classifier]
    print("Chosen classifier: " + detector)
    print("SF={0}, minN={1}".format(scaleFactor, minNeighbors))

    cat_cascade = cv.CascadeClassifier(cascade_models_dir + detector)
    eye_cascade = cv.CascadeClassifier(eye_cascade_model)

    if cat_cascade.empty():
        raise RuntimeError('The face classifier was not loaded correctly!')

    if eye_cascade.empty():
        raise RuntimeError('The eye classifier was not loaded correctly!')

    img = cv.imread(file)

    img_orig = cv.imread(file)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face = cat_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    if classifier == 0:
        col = (255, 0, 0)
    elif classifier == 1:
        col = (0, 255, 0)
    else:
        col = (0, 0, 255)

    cropped = None

    for (x, y, w, h) in face:  # blue
        img = cv.rectangle(img, (x, y), (x + w, y + h), col, 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray,
                                            scaleFactor=eyes_ScaleFactor,
                                            minNeighbors=eyes_minNeighbors,
                                            minSize=eyes_minSize)

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

        if len(eyes) == 0:
            print("No eyes detected")
        elif len(eyes) == 1:
            print("Only 1 eye (possibly) detected")
            cropped = img_orig[y:y + h, x: x + w]

        elif len(eyes) == 2:
            print("2 eyes detected!")
            cropped = img_orig[y:y + h, x: x + w]

            cropped = [cropped, eyes]
        else:
            print("More than 2 eyes (?) detected")
            cropped = img_orig[y:y + h, x: x + w]

    if show:
        cv.namedWindow('win', cv.WINDOW_NORMAL)
        # cv.resizeWindow('win', 1980, 1800)

        cv.imshow('win', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return cropped


def resize_image(img, width=None, height=None, inter=cv.INTER_AREA):
    """
    Resizes an image according to the specified parameters.

    :param img: image
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
    (h, w) = img.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return img

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
    resized = cv.resize(img, dim, interpolation=inter)

    # return the resized image
    return resized


def Distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def ScaleRotateTranslate(img, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return img.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)


def AlignFace(img, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.3, 0.3), dest_sz=(200, 200)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0 * offset_h
    # scale factor
    scale = float(dist) / float(reference)
    # rotate original around the left eye
    img = ScaleRotateTranslate(img, center=eye_left, angle=rotation)
    # crop the rotated image - [Not needed as the image is already cropped]
    # crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    # crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
    # img = img.crop(
    #     (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    # resize it
    img = img.resize(dest_sz, Image.ANTIALIAS)
    return img


def show_image(im):
    cv.namedWindow('win', cv.WINDOW_NORMAL)
    # cv.resizeWindow('win', 1980, 1800)

    cv.imshow('win', im)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    """Main for image cropping & testing purposes"""

    out_dir = "../images/dataset/cropped/"

    image = "../images/dataset/unprocessed/c/c10.jpeg"
    split = image.split("/")
    dir_name = split[-2]
    file_name = split[-1].split(".")[0]
    file_extension = split[-1].split(".")[1]

    save_dir = out_dir + dir_name + "/"

    out = detect_cat_face(image, classifier=2, show=True, scaleFactor=1.02, minNeighbors=2)
    if len(out) == 2:
        face = out[0]
        # show_image(img)

        # transform image into a PIL Image (for face Alignment)
        trans = cv.cvtColor(face, cv.COLOR_BGR2RGB)
        im_pil = Image.fromarray(trans)

        eye1 = out[1][0]
        eye2 = out[1][1]
        # print("eye1 = {0} -- eye2 = {1}".format(eye1, eye2))
        left_eye = np.minimum(eye1, eye2)
        right_eye = eye2 if np.array_equal(left_eye, eye1) else eye1
        # print(left_eye)
        # print(right_eye)

        im = AlignFace(im_pil,
                       eye_left=(int(left_eye[0]), int(left_eye[1])),
                       eye_right=(int(right_eye[0]), int(right_eye[1])))

        # im.show()
        # show_image(face)

        im_np = np.asarray(im_pil)

        cv.imwrite(save_dir + file_name + "_cropped" + ".jpg", face)
        im.save(save_dir + file_name + "_cropped_aligned." + file_extension, file_extension.upper())
