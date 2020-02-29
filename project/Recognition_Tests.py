from argparse import ArgumentParser
import cv2.cv2 as cv
from itertools import product
import json
import numpy as np
import random
import os

import Recognizer as rec
import Eyes_Recognizer as eye_rec
import utils


"""
def k_fold_cross_validation(dataset_path, k=10, tot_subjects=23):
    '''
    Generates all possible combinations of k subsets
    from the original dataset.

    :param dataset_path: path to the dataset file
    :param k: the number of subsets to generate
    :param tot_subjects: the total number of subjects inside the dataset
    :return:
    '''
    subjects = open(dataset_path, 'r')
    dataset = []  # [[all images of subject 1][all images of subject 2]....]
    new_list = []

    cont = 1
    ''' Create a list of lists, which are the images of different subjects '''
    for line in subjects:
        image_and_class = line.split(';')
        image_class = image_and_class[1]
        num_class = image_class.split('\n')[0]
        if int(num_class) == cont:
            new_list.append(line)
        else:
            dataset.append(new_list)
            cont += 1
            new_list = []
            new_list.append(line)
            if int(num_class) == tot_subjects:
                dataset.append(new_list)
    # print(dataset)
    subjects.close()

    ''' Create a dictionary of subject:number of images of the subject '''
    d = dict()
    classes = 1
    for cl in dataset:
        for subj in cl:
            d[classes] = len(cl)
        classes += 1
    # print(d)

    ''' Split the dataset into k parts '''
    subj_list = []

    for c in d:
        num_subj_for_subs = math.floor(d[c] / k)  # number of subject images in each subset
        subj_list.append(num_subj_for_subs)
    # print('-----------')
    # print(subj_list)

    ''' Creation of a list containing all the subsets with different images for the subjects '''
    all_subsets = []
    n_list = []
    c = -1
    cont_elem_subj = 0

    for i in range(k):
        for sub in dataset:
            c += 1
            n_elem = 0
            cont_elem_subj = i * subj_list[c]
            while n_elem < subj_list[c]:
                n_list.append(sub[cont_elem_subj])
                n_elem += 1
                cont_elem_subj += 1
        all_subsets.append(n_list)
        n_list = []
        c = -1

    # print(all_subsets)
    # print('-------')

    ''' Creation of (k-1)subsets for training and 1 for test '''
    final_list = []
    for i in range(k):
        train = all_subsets.copy()
        test = all_subsets[i]
        del train[i]
        couple = (train, test)
        final_list.append(couple)

    ret = []

    ''' Join all the sub-lists in the train element'''
    for f in final_list:
        tr = set()
        te = set()

        for ll in f[0]:
            for lll in ll:
                if "\n" not in lll:
                    tr.add(lll + "\n")
                else:
                    tr.add(lll)
        for cc in f[1]:
            te.add(cc)

        ret.append((tr, te))

        # print(len(train))
        # # print(train)
        # print(len(test))
        # # print(test)

    return ret
"""


def k_fold_cross_validation(dataset_path, k=5, n_impostors=1):
    """
    Generates all possible combinations of k subsets
    from the original dataset.

    :param dataset_path: path to the dataset file
    :param k: the number of subsets to generate
    :param n_impostors: the number of impostors to use
    :return:
    """
    label_to_file, files = utils.read_csv(dataset_path, mapping=True)
    mn = min(len(x) for x in label_to_file.values())

    pps = int(mn / k)
    st = 0

    ls = list(label_to_file.values())

    impostors = list(random.choices(list(label_to_file.keys()), k=k * n_impostors))
    # print(impostors)

    for li in ls:
        # shuffle the lists in order to get different results for each run of the function
        random.shuffle(li)

    subsets = []
    for i in range(k):
        s = set()

        for subj_lst in ls:
            s.update(subj_lst[st:st + pps])

        subsets.append(s)

        st += pps

    ret = []
    for i in range(k):
        # each time, the i-th element is the testing subset
        test = subsets[i]

        # and use the rest as training subset
        training = set()

        cpy = subsets[:]
        cpy.remove(test)

        for c in cpy:
            training.update(c)

        imps = impostors[i * n_impostors:(i + 1) * n_impostors]
        for imp in imps:
            for image in list(training)[:]:
                if utils.get_label(image) == imp:
                    training.remove(image)

        # ret.append((list(training), list(test)))

        training_ls = list(training)
        testing_ls = list(test)

        training_ls = [x+";"+str(utils.get_label(x)) for x in training_ls]
        testing_ls = [x+";"+str(utils.get_label(x)) for x in testing_ls]

        ret.append((training_ls, testing_ls))

    return ret


def compute_distance_matrix(test_csv, resize, model, height, use_eyes=False):
    """
    Creates an all-against-all (probes vs  gallery)
    distance matrix for identification.

    :param test_csv: file containing the probe images paths
    :param resize: flag to resize the probe images or not
    :param model: recongizer to be used
    :param height: height of each photo
    :return: generated distance matrix
    """

    # print("Creating distance matrix...")

    matrix = dict()
    # matrix_ser = dict()  # matrix to be serialized

    label_to_file, files = utils.read_csv(test_csv, resize=resize, mapping=True)

    probe_labels = set()
    for file in files:
        label = utils.get_label(file)
        probe_labels.add(label)

        if not use_eyes:
            prediction = rec.predict(model=model, height=height, resize=resize,
                                    probe_label=label, probe_image=file, identification=True)
        else:
            prediction = eye_rec.predict(model=model, height=height, resize=resize,
                                    probe_label=label, probe_image=file, identification=True)

        matrix[(file, label)] = prediction

        # matrix_ser["{0}#{1}".format(file, label)] = prediction

    # print("Matrix computed. Trying to serialize...")
    # import datetime
    # serialize_matrix(matrix_ser, '../test/1/matrix{}.json'.format(datetime.datetime.now()))

    return matrix  # , probe_labels


def evaluate_performances(model, thresholds, train_csv, test_csv, resize=True, use_eyes=False):
    """
    Compute FAR, FRR, GRR and DIR(k) for each threshold passed in input
    based on the couple of training and testing files provided.

    :param model: model to be used
    :param thresholds: thresholds to test
    :param train_csv: file containing the images to be used for training
    :param test_csv: file containing the images to be used for testing
    :param resize: flag to resize the images
    :return: dictionary containing the computed rates
    """

    # print("Evaluating performances for files {} {}...\n".format(train_csv, test_csv))

    model, height, gallery_labels = rec.train_recongizer(model, train_csv, resize, ret_labels=True)
    # print(gallery_labels)

    distance_matrix = compute_distance_matrix(test_csv, resize, model=model, height=height, use_eyes=use_eyes)

    # print("\nStarting performances computation...")
    all_probes = list(distance_matrix.keys())

    genuine_labels = [x[1] for x in all_probes if x[1] in gallery_labels]
    genuine_attempts = len(genuine_labels)
    impostors_labels = [x[1] for x in all_probes if x[1] not in gallery_labels]
    impostor_attempts = len(impostors_labels)

    # print('Impostors: ', impostor_attempts, impostors_labels, set(impostors_labels))
    # print('Genuines: ', genuine_attempts, genuine_labels, set(genuine_labels))

    performances = dict()

    for t in thresholds:
        fa = 0  # False accepts counter
        fr = 0  # False rejects counter -- Not used but still kept track of
        gr = 0  # Genuine rejects counter
        di = dict()  # Correct detect and identification @ rank k counter
        di[1] = 0
        for probe in all_probes:
            probe_label = probe[1]

            results = distance_matrix[probe]

            first_result = results[0]
            fr_label = first_result[0]
            fr_distance = first_result[1]

            # Impostor attempt
            if probe_label in impostors_labels:
                if fr_distance <= t:
                    fa += 1
                else:
                    gr += 1

            # Check if a correct identification @ rank 1 happened
            elif fr_label == probe_label:
                # Check if distance is less than the threshold
                if fr_distance <= t:
                    di[1] += 1
                else:
                    fr += 1

            # Find the first index (rank) in results where a correct match happens
            else:
                for res in results:
                    if res[0] == probe_label:
                        ind = results.index(res)
                        di[ind] = di[ind] + 1 if ind in di.keys() else 1

                        break

        # write_scores(dir1scores)

        # Compute rates
        dir_k = dict()  # Correct detect & identify rate @ rank k
        dir_k[1] = di[1] / genuine_attempts
        frr = 1 - dir_k[1]
        far = fa / impostor_attempts
        grr = gr / impostor_attempts

        higher_ranks = sorted(list(di.keys()))
        higher_ranks.remove(1)  # remove first rank, as here we're interested in the higher ones
        for k in higher_ranks:
            if k-1 not in dir_k.keys():
                dir_k[k - 1] = dir_k[max(dir_k.keys())]
            dir_k[k] = (di[k] / genuine_attempts) + dir_k[k - 1]

        performances[t] = dict([("FRR", frr), ("FAR", far), ("GRR", grr), ("DIR", dir_k)])

    # print(performances)
    # print("Done\n--\n")

    return performances


def serialize_matrix(matrix, out_file):
    with open(out_file, "w+") as fi:
        ob = json.dumps(matrix)
        fi.write(ob)


def load_matrix(file):
    # TODO read & reformat matrix according to the real one
    with open(file, "r+") as fi:
        return json.loads(fi.read())


def evaluate_avg_performances(recognizer, thresholds, files, use_eyes=False):
    """
    Computes averages of what is generated
    by the evaluate_performances() function.

    :param recognizer: model to be used
    :param thresholds: chosen thresholds
    :param files: iterable containing couples of training and testing files
    :return: dictionary with average rates
    """
    # print("Starting to compute performances...")

    avg_performances_per_threshold = dict()

    for threshold in thresholds:
        avg_performances_per_threshold[threshold] = dict([("AVG_FRR", 0), ("AVG_FAR", 0), ("AVG_GRR", 0),
                                                          ("AVG_DIR", dict())])

    for train_f, test_f in files:
        # Returns a dictionary "Threshold: rates for the threshold" based on the 'train' & 'test' files
        perf = evaluate_performances(model=recognizer, thresholds=thresholds, train_csv=train_f, test_csv=test_f, use_eyes=use_eyes)

        for threshold in thresholds:
            avg_performances_per_threshold[threshold]["AVG_FRR"] += perf[threshold]["FRR"]
            avg_performances_per_threshold[threshold]["AVG_FAR"] += perf[threshold]["FAR"]
            avg_performances_per_threshold[threshold]["AVG_GRR"] += perf[threshold]["GRR"]

            for k in perf[threshold]["DIR"]:
                if k not in avg_performances_per_threshold[threshold]["AVG_DIR"].keys():
                    avg_performances_per_threshold[threshold]["AVG_DIR"][k] = perf[threshold]["DIR"][k]
                else:
                    avg_performances_per_threshold[threshold]["AVG_DIR"][k] += perf[threshold]["DIR"][k]

    # print("Finishing averages computation...")

    for threshold in thresholds:
        avg_performances_per_threshold[threshold]["AVG_FRR"] /= len(files)
        avg_performances_per_threshold[threshold]["AVG_FAR"] /= len(files)
        avg_performances_per_threshold[threshold]["AVG_GRR"] /= len(files)

        for k in avg_performances_per_threshold[threshold]["AVG_DIR"].keys():
            avg_performances_per_threshold[threshold]["AVG_DIR"][k] /= len(files)

    # print("Averages:\n\t")
    # print(avg_performances_per_threshold)
    # print("End")

    return avg_performances_per_threshold


# def write_scores(scores):
#     avg = sum(scores) / len(scores)
#     with open(test_files_folder+"dd.txt", 'a+') as fi:
#         fi.write("AVG: " + str(avg) + "\n")
#         fi.write(str(scores))
#         fi.write("\n------\n\n")


# def write_impostors(fd):
#     impostors_ltf, impostors_files = utils.read_csv(impostors_csv, mapping=True)
#
#     for imp_file in impostors_files:
#         fd.write(imp_file+";"+str(utils.get_label(imp_file))+"\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input_dataset', help='The path of the input dataset')
    parser.add_argument('-o', '--output', help='The path of the output directory', default='../test/k_fold/complete')
    parser.add_argument('-k', '--subsets', help='The number of subsets in which to divide the dataset', type=int, default=5)
    parser.add_argument('-i', '--impostors', help='The number of impostors to use', type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    subsets = args.subsets
    test_files_folder = os.path.join(args.output, 'csv')
    k_fold_files = list()

    # Gerenerating/loading k-fold subsets

    if os.path.exists(test_files_folder) and len(os.listdir(test_files_folder)) != 0:
        for i in range(subsets):
            train_fn = os.path.join(test_files_folder, "{}_train.csv".format(i + 1))
            test_fn = os.path.join(test_files_folder, "{}_test.csv".format(i + 1))

            k_fold_files.append((train_fn, test_fn))
    else:
        k_fold = k_fold_cross_validation(args.input_dataset, k=subsets, n_impostors=args.impostors)

        for i in range(len(k_fold)):
            train, test = k_fold[i]

            train_fn = os.path.join(test_files_folder, "{}_train.csv".format(i + 1))
            with open(train_fn, 'w+') as fi:
                fi.writelines("\n".join(train))

            test_fn = os.path.join(test_files_folder, "{}_test.csv".format(i + 1))
            with open(test_fn, 'w+') as fi:
                fi.writelines("\n".join(test))

            k_fold_files.append((train_fn, test_fn))

    print('=' * 80)
    print('K fold cross vaidation using k = {} subsets and {} impostors'.format(subsets, args.impostors))
    print('=' * 80)

    print('\n' + '-' * 80)
    print('Eigenfaces')
    print('-' * 80)

    default_components = 10000  # h * w
    n_components = [10, 80, default_components // 10, default_components]

    test_thresholds = np.linspace(1000, 5000, 100)

    avgs = list()
    model_names = list()

    for nc in n_components:
        print('Evaluating model #{} out of 4...'.format(len(avgs) + 1))

        face_recognizer = cv.face.EigenFaceRecognizer_create(num_components=nc)
        model_names.append('Eig. with {} comp'.format(nc))

        avgs.append(evaluate_avg_performances(face_recognizer, test_thresholds, k_fold_files))

    print('Done\n')

    utils.plot_error_rates(avgs, model_names)
    utils.plot_rocs(avgs, model_names)

    print('\n' + '-' * 80)
    print('Fisherfaces')
    print('-' * 80)

    test_thresholds = np.linspace(100, 1500, 100)

    avgs = list()
    model_names = list()

    for nc in n_components:
        print('Evaluating model #{} out of 4...'.format(len(avgs) + 1))

        face_recognizer = cv.face.FisherFaceRecognizer_create(num_components=nc)
        model_names.append('Fisher with {} comp'.format(nc))

        avgs.append(evaluate_avg_performances(face_recognizer, test_thresholds, k_fold_files))

    print('Done\n')

    utils.plot_error_rates(avgs, model_names)
    utils.plot_rocs(avgs, model_names)

    print('\n' + '-' * 80)
    print('LBPH')
    print('-' * 80)

    radius = [1, 2]
    neighbors = [4, 8, 12, 16]
    grid = [4, 8]
    models_tot = 24

    test_thresholds = np.linspace(1, 200, 100)

    avgs = list()
    model_names = list()

    for r, n, g in product(radius, neighbors, grid):
        if r == 1 and n > 8:
            continue

        print('Evaluating model #{} out of 24...'.format(len(avgs) + 1))

        face_recognizer = cv.face.LBPHFaceRecognizer_create(radius=r, neighbors=n, grid_x=g, grid_y=g)
        model_names.append('LBPH with radius {}, {} neighs, {}x{} grid'.format(r, n, g, g))

        avgs.append(evaluate_avg_performances(face_recognizer, test_thresholds, k_fold_files))

    print('Done\n')

    utils.plot_error_rates(avgs, model_names)
    utils.plot_rocs(avgs, model_names)


"""
    ''' Initialize recognizer and thresholds to be tested '''
    face_recognizer: cv.face_BasicFaceRecognizer = cv.face.EigenFaceRecognizer_create()
    # face_recognizer: cv.face_BasicFaceRecognizer = cv.face.FisherFaceRecognizer_create()
    # face_recognizer: cv.face_BasicFaceRecognizer = cv.face.LBPHFaceRecognizer_create()
    # test_thresholds = [t for t in range(3000, 10000, 100)]  # TODO SET ACCORDINGLY
    test_thresholds = [3045, 3200, 3500, 3700, 4000]

    # import sys
    # test_thresholds = [sys.maxsize]

    ''' Perform the k fold cross validation technique over the dataset'''
    dataset_file_path = '../dataset_info/subjects.csv'
    subsets = 4

    k_fold_files = []  # List of the path names of all the generated k_fold <train, test> couples

    test_files_folder = '../test/1/csv/'

    # Reload the k fold files if they were generated previously
    if os.path.exists(test_files_folder) and len(os.listdir(test_files_folder)) != 0:
        print("Loading pre-generated k fold files...")
        # for file in os.listdir(test_files_folder):
        for i in range(subsets):
            train_fn = test_files_folder + "{}_train.csv".format(i + 1)
            test_fn = test_files_folder + "{}_test.csv".format(i + 1)

            k_fold_files.append((train_fn, test_fn))

    else:
        print("Generating k-fold files...")
        k_fold = k_fold_cross_validation(dataset_file_path, k=subsets, n_impostors=3)

        for i in range(len(k_fold)):
            train, test = k_fold[i]

            train_fn = test_files_folder + "{}_train.csv".format(i + 1)
            with open(train_fn, 'w+') as fi:
                # fi.writelines(train)
                fi.writelines("\n".join(train))

            test_fn = test_files_folder + "{}_test.csv".format(i + 1)
            with open(test_fn, 'w+') as fi:
                # write_impostors(fi)  # Write the impostors files to the test csv
                # fi.writelines(test)
                fi.writelines("\n".join(test))

            k_fold_files.append((train_fn, test_fn))

    ''' Compute performances '''
    avg_per_threshold = evaluate_avg_performances(face_recognizer, test_thresholds, k_fold_files)

    utils.plot_error_rates([avg_per_threshold], ['test'])
    utils.plot_rocs([avg_per_threshold], ['test'])
"""
