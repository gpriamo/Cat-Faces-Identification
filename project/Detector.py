from argparse import ArgumentParser
import cv2.cv2 as cv
import math
import numpy as np
from os import path
from PIL import Image

from utils import *


# def_SF = 1.05  # play around with it (i.e. 1.05, 1.3 etc) Good ones: 1.04 (haar), 1.05
# def_N = 2  # play around with it (3,4,5,6) Good ones: 2 (haar)

cascade_models_dir = '../models/detection/'
cat_cascades = ['haarcascade_frontalcatface.xml', 'haarcascade_frontalcatface_extended.xml',
                'lbpcascade_frontalcatface.xml']
eye_cascade_model = path.join(cascade_models_dir, 'haarcascade_eye.xml')


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

    cat_cascade = cv.CascadeClassifier(path.join(cascade_models_dir, detector))
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
        show_image(img)

    return cropped


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


def Distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def AlignFace(img, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(200, 200)):
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
    # crop the rotated image
    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
    img = img.crop(
        (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    # resize it
    img = img.resize(dest_sz, Image.ANTIALIAS)
    return img


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input_image', help='The path of the input image')
    parser.add_argument('-o', '--output', help='The path of the output directory', default='../images/dataset/cropped/')
    parser.add_argument('-d', '--detector', default=0, type=int)
    parser.add_argument('-s', '--scalefactor', default=1.05, type=float)
    parser.add_argument('-n', '--minneighbors', default=2, type=int)
    parser.add_argument('-es', '--eyes-scalefactor', default=1.08, type=float)
    parser.add_argument('-en', '--eyes-minneighbors', default=3, type=int)
    parser.add_argument('-em', '--eyes-minsize', default=40, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    """Main for image cropping & testing purposes"""
    args = parse_args()

    out_dir = args.output
    image = args.input_image

    # TODO Remove
    # split = image.split("/")
    # dir_name = split[-2]
    # file_name = split[-1].split(".")[0]
    # file_extension = split[-1].split(".")[1]
    #
    # save_dir = out_dir + dir_name + "/"

    dir, file = path.split(image)
    dir_name = path.basename(dir)
    file_name, file_extension = path.splitext(file)

    save_dir = path.join(out_dir, dir_name)

    detector = args.detector
    sf = args.scalefactor
    n = args.minneighbors
    eyes_sf = args.eyes_scalefactor
    eyes_n = args.eyes_minneighbors
    eyes_ms = (args.eyes_minsize, args.eyes_minsize)

    out = detect_cat_face(image, classifier=detector, show=True, scaleFactor=sf, minNeighbors=n,
            eyes_ScaleFactor=eyes_sf, eyes_minNeighbors=eyes_n, eyes_minSize=eyes_ms)
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

        cv.imwrite(path.join(save_dir, file_name + "_cropped" + file_extension), face)
        im.save(path.join(save_dir, file_name + "_cropped_aligned" + file_extension))
