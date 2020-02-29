from argparse import ArgumentParser
import numpy as np
import cv2.cv2 as cv
import os
import utils


def norm_0_255(source: np.ndarray):
    """
    Normalizes an image.
    :param source: image to normalize.
    :return: normalized image.
    """
    src = source.copy()
    dst = None

    shape = source.shape
    if len(shape) == 2:  # single channel
        dst = cv.normalize(src, dst, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    elif len(shape) > 2 and shape[2] == 3:  # 3 channels
        dst = cv.normalize(src, dst, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
    else:
        dst = src.copy()

    return dst


def train_recongizer(model: cv.face_BasicFaceRecognizer, csv_filename, resize=True, ret_labels=False,
            save_dir=None,
            show_mean=False,
            save_mean=False,
            show_faces=False,
            save_faces=False):
    """
    Trains a face recognizer.

    :param model: face recognizer to be trained.
    :param csv_filename: file containing the images to be used for the training process.
    :param resize: flag to specify whether images should be resized.
    :param ret_labels: flag to specify whether the list of labels used for training should be returned.
    :param save_dir: directory where results should be saved.
    :param show_mean: if True, the mean image is shown.
    :param save_mean: if True, the mean image is saved in save_dir.
    :param show_faces: if True, eigenfaces/fisherfaces are shown.
    :param save_faces: if True, eigenfaces/fisherfaces are saved in save_dir.
    :return: the trained model and the height of the images used for the training.
    """
    faces, labels = utils.read_csv(csv_filename, resize)

    #  print("Total faces: {0}\nTotal labels: {1}".format(len(faces), len(set(labels))))

    height = faces[0].shape[0]

    model.train(faces, np.array(labels))

    # print("Train finished")

    if type(model) is cv.face_LBPHFaceRecognizer:
        if show_mean or show_faces:
            print("Model Information:")
            model_info = "\tLBPH(radius={}, neighbors={}, grid_x={}, grid_y={}, threshold={})".format(
                model.getRadius(),
                model.getNeighbors(),
                model.getGridX(),
                model.getGridY(),
                model.getThreshold())
            print(model_info)

            histograms = model.getHistograms()
            print("Size of the histograms: " + str(histograms[0].size))

    else:
        if show_mean or save_mean:
            mean = model.getMean()

            reshaped_mean = mean.reshape(height, -1)
            normalized_mean = norm_0_255(reshaped_mean)

            if show_mean:
                utils.show_image(normalized_mean)
            elif save_mean and save_dir is not None:
                cv.imwrite(os.path.join(save_dir, "mean.png"), normalized_mean)

        if show_faces or save_faces:
            eigenvalues: np.ndarray = model.getEigenValues()
            eigenvectors: np.ndarray = model.getEigenVectors()

            colormap = cv.COLORMAP_JET if type(model) is cv.face_EigenFaceRecognizer else cv.COLORMAP_BONE
            faces = []

            for i in range(0, min(10, len(eigenvectors.T))):
                msg = "Eigenvalue #{0} = {1}".format(i, eigenvalues.item(i))
                print(msg)

                ev = eigenvectors[:, i].copy()
                grayscale = norm_0_255(ev.reshape(height, -1))
                cgrayscale = cv.applyColorMap(grayscale, colormap)

                if show_faces:
                    faces.append(cgrayscale)
                elif save_faces and save_dir is not None:
                    file_name = os.path.join(save_dir, "eigenface_{}.png".format(i))
                    cv.imwrite(file_name, norm_0_255(cgrayscale))

            if show_faces:
                utils.show_images(faces)

    if ret_labels:
        return model, height, set(labels)

    return model, height


def predict(model: cv.face_BasicFaceRecognizer, height, probe_image, probe_label=None, resize=True, identification=True):
    """
    Performs a face recognition operation.

    :param model: face recognizer.
    :param height: height of the images used to train the model.
    :param probe_image: path to the image of the probe.
    :param probe_label: label of the probe.
    :param resize: flag to specify whether the probe image should be resized.
    :param identification: flag to specify the recognition operation
                           to carry out (True: identification, False: verification)
    :return: the result of the prediction.
    """
    if not os.path.exists(probe_image):
        raise RuntimeError("File {} does not exist!".format(probe_image))

    input_face = cv.imread(probe_image, 0)

    if resize:
        input_face = utils.resize_image(input_face, 100, 100)

    if identification:
        coll: cv.face_StandardCollector = cv.face.StandardCollector_create()
        pred = model.predict_collect(input_face, coll)
        # print(coll.getResults())
        # print(coll.getMinDist())
        # print(coll.getMinLabel())

        results = sorted(coll.getResults(), key=lambda x: x[1])

        # print(results)

        # if probe_label is not None:
        #     print("Predicted class = {0} ({1}) with confidence = {2}; Actual class = {3} ({4}).\n\t Outcome: {5}"
        #           .format(coll.getMinLabel(), get_subject_name(coll.getMinLabel()), coll.getMinDist(),
        #                   probe_label, get_subject_name(probe_label),
        #                   "Success!" if coll.getMinLabel() == probe_label else "Failure!"))

        return results

    prediction = model.predict(input_face)

    if probe_label is not None:
        print("Predicted class = {0} ({1}) with confidence = {2}; Actual class = {3} ({4}).\n\t Outcome: {5}"
              .format(prediction[0], utils.get_subject_name(prediction[0]), prediction[1],
                      probe_label, utils.get_subject_name(probe_label),
                      "Success!" if prediction[0] == probe_label else "Failure!"))


def save_model(model: cv.face_BasicFaceRecognizer, save_dir, height, uid=0):
    """
    Saves a recognizer model to file.
    :param model: model to be saved.
    :param save_dir: path where the model should be saved.
    :param height: height of the images used for the training.
    :param uid: identifier of the model to save.
    """
    file_name = os.path.join(save_dir, "model_{0}_{1}.xml".format(uid, height))
    print("Saving model to: ", file_name)
    model.save(file_name)


def load_model(model: cv.face_BasicFaceRecognizer, file_name):
    """
    Loads a previously-saved model from file.
    :param model: empty model the file should be loaded into.
    :param file_name: the file where the model is stored.
    :return: the loaded model.
    """
    model.read(file_name)
    height = file_name.split("_")[-1].split(".")[0]

    return model, int(height)


# def test_cropped(model: cv.face_BasicFaceRecognizer):
#     mod, hei = train_recongizer(model, "../dataset_info/bak/best/subjects.csv", resize=True)
#     predict(model=mod, height=hei, resize=True, probe_image="../images/dataset/cropped/s1/27.jpg", probe_label=1, identification=False)
#     predict(model=mod, height=hei, resize=True, probe_image="../images/dataset/cropped/s2/10.jpg", probe_label=2, identification=False)
#     predict(model=mod, height=hei, resize=True, probe_image="../images/dataset/cropped/s8/22.jpg", probe_label=8, identification=False)
#     save_model(mod, hei)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input_dataset', help='The path of the input dataset')
    parser.add_argument('-r', '--recognizer', help='The recognizer to use', type=int, choices=range(3), required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.recognizer == 0:
        model: cv.face_BasicFaceRecognizer = cv.face.EigenFaceRecognizer_create(num_components=10)

    elif args.recognizer == 1:
        model: cv.face_BasicFaceRecognizer = cv.face.FisherFaceRecognizer_create(num_components=80)

    elif args.recognizer == 2:
        model: cv.face_BasicFaceRecognizer = cv.face.LBPHFaceRecognizer_create(radius=2, neighbors=16)

    mod, hei = train_recongizer(model, args.input_dataset, show_mean=True, show_faces=True)

    # test_cropped(model)
