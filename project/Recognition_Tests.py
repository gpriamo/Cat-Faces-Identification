from project import Recognizer as rec
from project import utils
import cv2.cv2 as cv


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


def evaluate_performances(model, thresholds, resize=False):
    train_csv, test_csv = k_fold_cross_validation()
    model, height = rec.train_recongizer(model, train_csv, resize)

    distance_matrix = create_distance_matrix(test_csv, resize, model=model, height=height)
    # TODO for each threshold t: compute FAR(t), FRR(t), DIR(t, k), ...


if __name__ == '__main__':
    model: cv.face_BasicFaceRecognizer = cv.face.EigenFaceRecognizer_create()
    thresholds = [1.0, 2.0]

    evaluate_performances(model, thresholds=thresholds, resize=True)