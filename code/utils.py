import cv2.cv2 as cv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path

from ext.intersection import intersection


subject_to_name_file = '../dataset_info/subject-to-name.txt'
subject_to_name = None
dataset_images_dir = '../images/dataset/cropped'
# dataset_images_dir = '../images/dataset/best'
# dataset_images_dir = '../images/dataset/best_aligned'
dataset_info_dir = '../dataset_info/'


def show_image(im, matplot=True):
    """
    Shows a single image.
    :param im: image to show.
    :param matplot: flag to select the library to be used to display the image.
    """
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
    """
    Shows a set of images in the same window.
    :param images: images to show.
    """
    if len(images) < 4:
        size = (len(images), 1)
    else:
        size = (math.ceil(len(images) / 2), 2)

    fig = plt.figure(figsize=size)
    c, r = size

    for i, im in enumerate(images):
        fig.add_subplot(r, c, i + 1)
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
    separator = ";"

    print("Creating CSV file...")

    lines = []
    # lines_aligned = []

    for dir_name, dir_names, file_names in os.walk(base_path):
        dir_names.sort(key=lambda l: int(l.replace('s', '')))
        for subdir_name in dir_names:

            label = int(subdir_name.replace('s', ''))

            if subdir_name == "test":
                continue

            subject_path = path.join(dir_name, subdir_name)
            for filename in os.listdir(subject_path):

                if not path.isfile(path.join(subject_path, filename)):
                    continue

                abs_path = "%s/%s" % (subject_path, filename)
                s = "%s%s%d" % (abs_path, separator, label)
                # if "aligned" in s:
                #     # lines_aligned.append(s)
                #     pass
                # else:
                #     lines.append(s)
                lines.append(s)
            # label = label + 1

    # lines_aligned.sort(key=lambda l: (int(l.split("/")[-2].replace('s', '')), int(l.split("/")[-1].split("_")[0])))
    lines.sort(key=lambda l: (int(l.split("/")[-2].replace('s', '')), int(l.split("/")[-1].split(".")[0])))

    fname = "subjects.csv"
    # fname_al = "subjects_aligned.csv"

    # print(base_path)

    with open(path.join(output_dir, fname), "w+") as fl:
        fl.write(str.join("\n", lines))
        fl.write("\n")
    # with open(path.join(output_dir, fname_al), "w+") as fl:
    #     fl.write(str.join("\n", lines_aligned))
    #     fl.write("\n")


def read_csv(filename, resize=False, mapping=False):
    """
    Parses a csv file containing the path to the images to be used for the recognition operations.
    :param filename: location of the csv file.
    :param resize: flag to specify whether images
    :param mapping: flag to switch the output type.
    :return: if mapping = False: a list of loaded images alongside a list of their respective labels.
    If mapping = True: a list with just the files' names and a label -> filename mapping is returned.
    """
    labels = []
    faces = []

    with open(filename, "r+") as file:
        if mapping:
            label_to_file = dict()
            files = []

        dic = dict()

        for line in file.readlines():
            if line == "\n":
                break

            spl = line.split(";")

            im_file = spl[0]
            label = int(spl[1])

            if label not in dic.keys():
                dic[label] = 0
            # elif dic[label] >= 10:
            #     continue
            dic[label] += 1

            if mapping:
                if label not in label_to_file.keys():
                    label_to_file[label] = []
                label_to_file[label].append(im_file)

                files.append(im_file)

            else:
                photo = cv.imread(im_file, 0)

                if resize:
                    photo = resize_image(photo, 100, 100)

                faces.append(photo)
                labels.append(label)

    # print(dic)

    if mapping:
        return label_to_file, files

    return faces, labels


def _get_subject_mapping():
    """
    Maps the subjects' labels to their true names.
    """
    ret = dict()
    with open(subject_to_name_file, "r+") as fl:
        for line in fl.readlines():
            spl = line.split("\t")
            ret[spl[0]] = spl[1].replace("\n", "")
    return ret


def get_subject_name(label):
    """
    :return: the true name of the subject with the specified label
    """
    global subject_to_name

    if subject_to_name is None:
        subject_to_name = _get_subject_mapping()

    key = "s"+str(label)
    if key in subject_to_name:
        return subject_to_name[key]
    else:
        return "Impostor"


def get_label(file):
    """
    :return: a label from a file name.
    """
    return int(file.split("/")[-2].replace('s', ''))


def parse_identification_results(result):
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


def print_avg_performances(performances, threshold):
    print('SUMMARY at threshold {}:'.format(threshold))
    print('\tFAR:', performances[threshold]['AVG_FAR'])
    print('\tFRR:', performances[threshold]['AVG_FRR'])
    print('\tGRR:', performances[threshold]['AVG_GRR'])
    print('\tDIR at rank 1:', performances[threshold]['AVG_DIR'][1], '\n')


def plot_error_rates(performancies, model_names, normalize_threshols=False):
    plt.figure()

    for avg_per_threshold, model_name in zip(performancies, model_names):
        thresholds = list()
        fars = list()
        frrs = list()
        for t, performs in avg_per_threshold.items():
            thresholds.append(t)
            fars.append(performs['AVG_FAR'])
            frrs.append(performs['AVG_FRR'])

        if normalize_threshols:
            thresholds = np.array(thresholds)
            thresholds = (thresholds - np.min(thresholds)) / (np.max(thresholds) - np.min(thresholds))

        plt.plot(thresholds, fars, label=model_name + ': FAR')
        plt.plot(thresholds, frrs, label=model_name + ': FRR')

        err_t, err = intersection(np.array(thresholds), np.array(fars), np.array(thresholds), np.array(frrs))

        print('{}: ERR of {} reached at threshold {}'.format(model_name, err, err_t))

        if len(err) != 0:
            plt.scatter(err_t, err, color='gray')
            plt.axvline(x=err_t, color='gray', linestyle='--')
            plt.annotate('ERR', (err_t, err))

    plt.xlabel('Tolerance Threshold')
    plt.ylabel('Error Rate')
    plt.grid()
    plt.legend()
    plt.show()


def plot_rocs(performancies, model_names):
    plt.figure()
    plt.title('Watchlist ROC')

    for avg_per_threshold, model_name in zip(performancies, model_names):
        fars = list()
        dirs = list()
        for performs in avg_per_threshold.values():
            fars.append(performs['AVG_FAR'])
            dirs.append(performs['AVG_DIR'][1])

        plt.plot(fars, dirs, linewidth=2, label=model_name)

    plt.xlabel('False Alarm Rate')
    plt.ylabel('Detect and Identify Rate')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    create_csv(dataset_images_dir, dataset_info_dir)
