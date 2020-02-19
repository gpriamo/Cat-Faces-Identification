import cv2.cv2 as cv
import math
import matplotlib.pyplot as plt
import os
from os import path


subject_to_name_file = '../dataset_info/subject-to-name.txt'
subject_to_name = None
dataset_images_dir = '../images/dataset/cropped'
dataset_info_dir = '../dataset_info/'


def show_image(im, matplot=True):
    if matplot:
        # show image using matplotlib
        plt.figure()
        plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        plt.show()

    else:
        # show image using opencv
        cv.namedWindow('win', cv.WINDOW_NORMAL)
        # cv.resizeWindow('win', 1980, 1800)

        cv.imshow('win', im)
        cv.waitKey(0)
        cv.destroyAllWindows()


def show_images(images):
    if len(images) < 4:
        size = (len(images), 1)
    else:
        size = (math.ceil(len(images) / 2), 2)

    fig = plt.figure(figsize=size)
    c, r = size

    for i, im in enumerate(images):
        fig.add_subplot(r, c, i+1)
        plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))

    plt.show()


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
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return img

    dim = (width, height)
    resized = cv.resize(img, dim, interpolation=inter)

    # return the resized image
    return resized


def create_csv(base_path, output_dir):
    """
    Creates the CSV file needed to train the recognizers.

    :param base_path:
        directory where all the training images are stored.
    :param output_dir:
        directory where to save CSV files.
    """
    label = 1
    separator = ";"

    print("Creating CSV file...")

    lines = []
    lines_aligned = []

    for dir_name, dir_names, file_names in os.walk(base_path):
        dir_names.sort(key=lambda l: int(l.replace('s', '')))
        for subdir_name in dir_names:

            if subdir_name == "test":
                continue

            subject_path = path.join(dir_name, subdir_name)
            for filename in os.listdir(subject_path):

                if not path.isfile(path.join(subject_path, filename)):
                    continue

                abs_path = "%s/%s" % (subject_path, filename)
                s = "%s%s%d" % (abs_path, separator, label)
                if "aligned" in s:
                    lines_aligned.append(s)
                else:
                    lines.append(s)
            label = label + 1

    lines_aligned.sort(key=lambda l: (int(l.split("/")[-2].replace('s', '')), int(l.split("/")[-1].split("_")[0])))
    lines.sort(key=lambda l: (int(l.split("/")[-2].replace('s', '')), int(l.split("/")[-1].split(".")[0])))

    with open(path.join(output_dir, "subjects.csv"), "w+") as fl:
        fl.write(str.join("\n", lines))
    with open(path.join(output_dir, "subjects_aligned.csv"), "w+") as fl:
        fl.write(str.join("\n", lines_aligned))


def read_csv(filename, resize=False, rgb=False, mapping=False):
    labels = []
    faces = []

    with open(filename, "r+") as file:
        if mapping:
            label_to_file = dict()
            files = []

        # {BEGIN} TOREMOVE
        dic = dict()
        # {END} TOREMOVE

        for line in file.readlines():
            spl = line.split(";")

            im_file = spl[0]
            label = int(spl[1])

            # {BEGIN} TOREMOVE
            if label not in dic.keys():
                dic[label] = 1
            elif dic[label] >= 10:
                continue
            dic[label] += 1
            # {END} TOREMOVE

            if mapping:
                if label not in label_to_file.keys():
                    label_to_file[label] = []
                label_to_file[label].append(im_file)

                files.append(im_file)

            else:
                photo = cv.imread(im_file, 0)

                if resize:
                    photo = resize_image(photo, 100, 100)

                if rgb:
                    # Convert input image from BGR to RGB (needed by dlib)
                    photo = cv.cvtColor(photo, cv.COLOR_BGR2RGB)

                faces.append(photo)
                labels.append(label)

    # {BEGIN} TOREMOVE
    print(dic)

    # for key in label_to_file.keys():
    #     print(key, label_to_file[key])
    # {END} TOREMOVE

    if mapping:
        return label_to_file, files

    return faces, labels


def _get_subject_mapping():
    ret = dict()
    with open(subject_to_name_file, "r+") as fl:
        for line in fl.readlines():
            spl = line.split("\t")
            ret[spl[0]] = spl[1].replace("\n", "")
    return ret


def get_subject_name(label):
    global subject_to_name

    if subject_to_name is None:
        subject_to_name = _get_subject_mapping()

    key = "s"+str(label)
    if key in subject_to_name:
        return subject_to_name[key]
    else:
        return "Impostor"


def get_label(file):
    return int(file.split("/")[-2].replace('s', ''))


def parse_identification_result(result):
    # ret = dict()
    #
    # for k in result:
    #     label = k[0]
    #     distance = k[1]
    #
    #     if label not in ret.keys():
    #         ret[label] = distance
    #     else:
    #         ret[label] = min(distance, ret[label])

    return sorted(list(dict(sorted(result, key=lambda x: int(x[1]), reverse=True)).items()), key=lambda x: int(x[1]))


if __name__ == '__main__':
    create_csv(dataset_images_dir, dataset_info_dir)