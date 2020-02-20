from project import Recognizer as rec
from project import utils
import cv2.cv2 as cv
import json


def k_fold_cross_validation():
    """
    Placeholder method
    :return:
    """
    s = set()
    return s


def create_distance_matrix(test_csv, resize, model, height):
    matrix = dict()

    label_to_file, files = utils.read_csv(test_csv, resize=resize, mapping=True)

    for file in files:
        label = utils.get_label(file)
        matrix[(file, label)] = rec.predict(model=model, height=height, resize=resize,
                                            probe_label=label, probe_image=file, identification=True)

    return matrix


def evaluate_performances(model, thresholds, train_csv, test_csv, resize=False):
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
    model, height = rec.train_recongizer(model, train_csv, resize)

    distance_matrix = create_distance_matrix(test_csv, resize, model=model, height=height)

    performances = dict()
    for t in thresholds:
        genuine_attempts = 0
        impostor_attempts = 0

        fa = 0  # False accepts counter
        fr = 0  # False rejects counter -- Not used but still kept track of
        gr = 0  # Genuine rejects counter
        di = dict()  # Correct detect and identification @ rank k counter
        di[1] = 0
        for probe in distance_matrix.keys():
            probe_label = probe[1]

            results = distance_matrix[probe]

            gallery_labels = {x[0] for x in results}

            first_result = results[0]
            fr_label = first_result[0]
            fr_distance = first_result[1]

            # Impostor attempt
            if fr_label not in gallery_labels:
                impostor_attempts += 1

                if fr_distance <= t:
                    fa += 1
                else:
                    gr += 1
                continue

            genuine_attempts += 1

            # Check if a correct identification @ rank 1 happened
            if first_result[0] == probe_label:
                # Check if distance is less than the threshold
                if fr_distance <= t:
                    di[1] += 1
                else:
                    fr += 1
                continue

            # Find first index (rank) where a correct identification occurred
            for k in range(1, len(results)):
                res = results[k]
                res_label = res[0]
                res_distance = res[1]

                # Match found at rank k
                if res_label == probe_label:
                    if res_distance <= t:
                        di[k] = di[k] + 1 if k in di.keys() else 1  # Correct detect & identify @ rank k
                    else:
                        fr += 1
                        # Stop searching as distances have gone beyond the threshold
                        break
                elif res_distance <= t:
                    fr += 1
                    continue
                elif res_distance > t:  # Just "else" might be enough
                    fr += 1
                    # Stop searching as distances have gone beyond the threshold
                    break

        # Compute rates
        dir_k = dict()  # Correct detect & identify rate @ rank k
        dir_k[1] = di[1] / genuine_attempts
        frr = 1 - dir_k[1]
        far = fa / impostor_attempts
        grr = gr / impostor_attempts

        higher_ranks = sorted(list(di.keys()))
        higher_ranks.remove(1)  # remove first rank, as here we're interested in the higher ones
        for k in higher_ranks:
            dir_k = (di[k] / genuine_attempts) + dir_k[k - 1]

        performances[t] = dict([("FRR", frr), ("FAR", far), ("GRR", grr), ("DIR", dir_k)])

    return performances


def serialize_matrix(matrix, out_file):
    # TODO test
    with open(out_file, "w+") as fi:
        ob = json.dumps(matrix)
        fi.write(ob)


def load_matrix(file):
    # TODO test
    with open(file, "r+") as fi:
        return json.loads(fi.read())


def evaluate_avg_performances(recognizer, thresholds, files):
    avg_performances_per_threshold = dict()

    for threshold in test_thresholds:
        avg_performances_per_threshold[threshold] = dict([("AVG_FRR", 0), ("AVG_FAR", 0), ("AVG_GRR", 0),
                                                          ("AVG_DIR", dict())])

    for train, test in files:
        # Returns a dictionary "Threshold: rates for the threshold" based on the 'train' & 'test' files
        perf = evaluate_performances(model=recognizer, thresholds=thresholds, resize=True,
                                     train_csv=train, test_csv=test)

        for threshold in test_thresholds:
            avg_performances_per_threshold[threshold]["AVG_FRR"] += perf[threshold]["FRR"]
            avg_performances_per_threshold[threshold]["AVG_FAR"] += perf[threshold]["FAR"]
            avg_performances_per_threshold[threshold]["AVG_GRR"] += perf[threshold]["GRR"]

            for k in perf[threshold]["DIR"]:
                if k not in avg_performances_per_threshold[threshold]["AVG_DIR"].keys():
                    avg_performances_per_threshold[threshold]["AVG_DIR"][k] = perf[threshold]["DIR"][k]
                else:
                    avg_performances_per_threshold[threshold]["AVG_DIR"][k] += perf[threshold]["DIR"][k]

    for threshold in test_thresholds:
        avg_performances_per_threshold[threshold]["AVG_FRR"] /= len(k_fold_files)
        avg_performances_per_threshold[threshold]["AVG_FAR"] /= len(k_fold_files)
        avg_performances_per_threshold[threshold]["AVG_GRR"] /= len(k_fold_files)

        for k in avg_performances_per_threshold[threshold]["AVG_DIR"].keys():
            avg_performances_per_threshold[threshold]["AVG_DIR"][k] /= len(k_fold_files)

    return avg_performances_per_threshold


if __name__ == '__main__':
    face_recognizer: cv.face_BasicFaceRecognizer = cv.face.EigenFaceRecognizer_create()
    test_thresholds = [1.0, 2.0]

    k_fold_files = k_fold_cross_validation()

    avg_per_threshold = evaluate_avg_performances(face_recognizer, test_thresholds, k_fold_files)
