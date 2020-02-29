from argparse import ArgumentParser
import cv2.cv2 as cv
import math
import numpy as np
from os import path
from PIL import Image
import glob
import shutil

from utils import *
from Recognizer import norm_0_255, train_recongizer



class_name = ["Blue", "Green", "Yellow", "Brown", "Gray"]


# rgb colors
eyes_color = {
    class_name[0] : [(150, 175, 195), (55, 111, 168)],  # crystalino, roxy  #
    class_name[1] : [(163, 163, 125), (71, 84, 74)],    # perla
    class_name[2] : [(238, 255, 172), (196, 170, 121)],    # bortolo 
    class_name[3] : [(130, 101, 0), (51, 41, 40)],
    class_name[4] : [(128, 128, 128), (90, 85, 79)]   # tigro
}



cascade_models_dir = '../models/detection/'
eye_cascade_model = path.join(cascade_models_dir, 'haarcascade_eye.xml')


def detect_cat_eyes(file, show=False, eyes_ScaleFactor=1.08, eyes_minNeighbors=3, eyes_minSize=(40, 40)):
    """
    Cat eyes detection utility.

    :param file : str
        The name of the image file to detect the face from.
    :param eyes_ScaleFactor: float
        scaleFactor value the eyes detector should use
    :param eyes_minNeighbors:
        minNeighbors value the eyes detector should use
    :param eyes_minSize:
        minSize value the eyes detector should use
    :return the cropped face and the location of the eyes, if detected, else None.
    """

    eye_cascade = cv.CascadeClassifier(eye_cascade_model)

    if eye_cascade.empty():
        raise RuntimeError('The eye classifier was not loaded correctly!')

    img = cv.imread(file)

    img_orig = cv.imread(file)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cropped = None

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=eyes_ScaleFactor,
                                              minNeighbors=eyes_minNeighbors,
                                              minSize=eyes_minSize)
    cont = 0

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    for (ex, ey, ew, eh) in eyes:
        cont += 1
        eye = cv.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
        if len(eyes) == 2:
            crop_eyes = eye[ey+2:ey + eh-1, ex+2: ex + ew-1]
            print('Saving eye image...')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cv.imwrite(path.join(save_dir, file_name + "_" + str(cont) + file_extension), crop_eyes)

        elif len(eyes) == 1:
            crop_eyes = eye[ey+2:ey + eh-1, ex+2: ex + ew-1]
            print('Saving eye image...')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cv.imwrite(path.join(save_dir, file_name + "_" + str(cont) + file_extension), crop_eyes)


    num_eyes = []
    for filename in glob.glob(save_dir+'/*'): 
        img = cv.imread(filename, cv.IMREAD_COLOR)
        num_eyes.append(img)
    #print(len(num_eyes))


    if len(eyes) == 2 or len(num_eyes) == 2: 
        print("2 eyes detected!")

        pixel_count_1, pixel_count_2 = analysis_color_eyes()

        possible_classes_1 = left_eye_color(pixel_count_1)
        possible_classes_2 = right_eye_color(pixel_count_2)
        print('CLASSI 1', possible_classes_1)
        print('CLASSI 2', possible_classes_2)

        color_eyes = final_eyes_color(possible_classes_1, possible_classes_2)
        print(color_eyes)

        subj_list = []
        subj_list = find_subject_from_eyes_color(color_eyes, subj_list)

        return subj_list
            
    elif len(eyes) == 1 or len(num_eyes) == 1:
        print("Only 1 eye detected!")

        pixel_count_1, pixel_count_2 = analysis_color_eyes()
        possible_classes_1 = left_eye_color(pixel_count_1)
        print('CLASSI 1', possible_classes_1)

        subj_list = []
        subj_list = find_subject_from_eyes_color(possible_classes_1, subj_list)

        return subj_list


    else:
        print("No eyes detected!")
        file_eyes = open('../dataset_info/gallery_eyes_color.txt')
        lines=[]
        for line in file_eyes:
            l = line.split('\n')[0]
            lines.append(l)
        #print(lines)

        cats = []
        for cat in lines:
            s = cat.split('  ')
            cats.append(s)

        return cats



def analysis_color_eyes():
    # Eyes color recognition
    eyes_pixel_list = []

    for filename in glob.glob(save_dir+"/*"): 
        img = cv.imread(filename, cv.IMREAD_COLOR)
        im = Image.open(filename)
        pix = im.load()
        height, width = img.shape[:2]

        #print(height,width)
        height=height-1
        width=width-1

        color_pixel_list = []
        for eh in range(height):
            for ew in range(width):
                r,g,b=pix[ew,eh]
                #print(r,g,b)
                if r<=30 and g<=50 and b<=114:
                    #print(eh,ew)
                    cv.circle(img,(ew,eh),1,(0,255,0),1)
                else:
                    triple = (r,g,b)
                    color_pixel_list.append(triple)
        eyes_pixel_list.append(color_pixel_list)

    #print(len(eyes_pixel_list))
    if len(eyes_pixel_list) == 2:

        pixel_class_1 = {}
        pixel_class_2 = {}
        cont_1 = 0
        cont_2 = 0
        for i in range(len(class_name)):
            pixel_class_1[class_name[i]] = 0
            pixel_class_2[class_name[i]] = 0

        for lista in eyes_pixel_list:
            for triple in lista:
                #print(triple)
                r,g,b = triple
                for i in range(len(class_name)):
                    color = eyes_color[class_name[i]]
                    if (triple[0] <= color[0][0] and (triple[0] >= color[1][0]) and (triple[1] <= color[0][1]) and triple[1] >= color[1][1] and (triple[2] <= color[0][2]) and (triple[2] >= color[1][2])):
                        if eyes_pixel_list.index(lista) == 0:
                            cont_1 += 1
                            pixel_class_1[class_name[i]] = cont_1
                        else:
                            cont_2 += 1
                            pixel_class_2[class_name[i]] = cont_2
                    #probe_eye_class = class_name[i]
                    #print(class_name[i])
                 #   print(triple)

        print(pixel_class_1)
        print('-------')
        print(pixel_class_2)

        pixel_count_1 = []
        pixel_count_2 = []
        for c in range(len(class_name)):
            pixel_count_1.append(pixel_class_1[class_name[c]])
            pixel_count_2.append(pixel_class_2[class_name[c]])

        return pixel_count_1, pixel_count_2

    if len(eyes_pixel_list) == 1:
        pixel_class_1 = {}
        cont_1 = 0
        for i in range(len(class_name)):
            pixel_class_1[class_name[i]] = 0

        for lista in eyes_pixel_list:
            for triple in lista:
                #print(triple)
                r,g,b = triple
                for i in range(len(class_name)):
                    color = eyes_color[class_name[i]]
                    if (triple[0] <= color[0][0] and (triple[0] >= color[1][0]) and (triple[1] <= color[0][1]) and triple[1] >= color[1][1] and (triple[2] <= color[0][2]) and (triple[2] >= color[1][2])):
                        if eyes_pixel_list.index(lista) == 0:
                            cont_1 += 1
                            pixel_class_1[class_name[i]] = cont_1
                    #probe_eye_class = class_name[i]
                    #print(class_name[i])
                 #   print(triple)

        print(pixel_class_1)

        pixel_count_1 = []
        for c in range(len(class_name)):
            pixel_count_1.append(pixel_class_1[class_name[c]])

        return pixel_count_1, []

        


def left_eye_color(pixel_count_1):
    possible_classes_1 = set()
    first = max(pixel_count_1)
    for i in pixel_count_1:
        if abs(i-first) > 50 and class_name[pixel_count_1.index(first)]: # and first==max(pixel_count_1):
            possible_classes_1.add(class_name[pixel_count_1.index(first)])
        elif abs(i-first) <= 50 and class_name[pixel_count_1.index(first)] != class_name[pixel_count_1.index(i)] and pixel_count_1.index(i)!= 0:
            if class_name[pixel_count_1.index(first)] in possible_classes_1:
                possible_classes_1.add(class_name[pixel_count_1.index(i)])
            else:
                possible_classes_1.add(class_name[pixel_count_1.index(first)])
                possible_classes_1.add(class_name[pixel_count_1.index(i)])
    return possible_classes_1



def right_eye_color(pixel_count_2):
    possible_classes_2 = set()
    first = max(pixel_count_2)
    for i in pixel_count_2:
        if abs(i-first) > 50 and class_name[pixel_count_2.index(first)]:
            possible_classes_2.add(class_name[pixel_count_2.index(first)])
        elif abs(i-first) <= 50 and class_name[pixel_count_2.index(first)] != class_name[pixel_count_2.index(i)] and pixel_count_2.index(i)!= 0:
            if class_name[pixel_count_2.index(first)] in possible_classes_2:
                possible_classes_2.add(class_name[pixel_count_2.index(i)])
            else:
                possible_classes_2.add(class_name[pixel_count_2.index(first)])
                possible_classes_2.add(class_name[pixel_count_2.index(i)])
    return possible_classes_2



def final_eyes_color(possible_classes_1, possible_classes_2):
    color_eyes = []
    s1 = set(possible_classes_1)
    s2 = set(possible_classes_2)

    if not s1.intersection(s2):
        color_eyes = ['Different']
    else:
        color_eyes = list(s1.intersection(s2))

    return color_eyes


def find_subject_from_eyes_color(color_eyes, subj_list):
    file_eyes = open('../dataset_info/gallery_eyes_color.txt')
    lines=[]
    for line in file_eyes:
        l = line.split('\n')[0]
        lines.append(l)
    #print(lines)

    cats = []
    for cat in lines:
        s = cat.split('  ')
        cats.append(s)
    #print(cats)

    for cat in cats:
        for col in color_eyes:
            if col == cat[2]:
                subj_list.append(int(cat[1][1:]))
    return subj_list



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_image', help='The path of the input image', default='../images/dataset/cropped/s13/9.jpg')   #default='../images/dataset/unprocessed/done/EYES/19_cropped.jpeg')
    parser.add_argument('-o', '--output', help='The path of the output directory', default='../images/dataset/Eyes/')   #default='../images/dataset/unprocessed/')
    parser.add_argument('-es', '--eyes-scalefactor', default=1.5, type=float) 
    parser.add_argument('-en', '--eyes-minneighbors', default=3, type=int)
    parser.add_argument('-em', '--eyes-minsize', default=40, type=int)
    parser.add_argument('-r', '--recognizer', help='The recognizer to use', type=int, choices=range(3), required=True)
    

    return parser.parse_args()



def predict(subj_list, model: cv.face_BasicFaceRecognizer, height, probe_image, probe_label=None, resize=False,
            identification=True,
            save_dir=None,
            show_mean=False,
            save_mean=False,
            show_faces=False,
            save_faces=False
            ):
    if not path.exists(probe_image):
        raise RuntimeError("File {} does not exist!".format(probe_image))

    input_face = cv.imread(probe_image, 0)

    if resize:
        input_face = resize_image(input_face, 100, 100)

    if identification:
        coll: cv.face_StandardCollector = cv.face.StandardCollector_create()
        pred = model.predict_collect(input_face, coll)

        results = sorted(coll.getResults(), key=lambda x: x[1])
        #for elem in results:
        #    print(elem)
        #print('LISTA SOGGETTI', len(subj_list))
        if len(subj_list) != 23:
            limited_subjects_list = []
            for t in results:
                sub, sco = t
                for i in subj_list:
                    if i == sub:
                        limited_subjects_list.append(t)

        #print(limited_subjects_list)


            if probe_label is not None:
                print("Predicted class = {0} ({1}) with confidence = {2}; Actual class = {3} ({4}).\n\t Outcome: {5}"
                      .format(limited_subjects_list[0][0], get_subject_name(limited_subjects_list[0][0]), limited_subjects_list[0][1],
                              probe_label, get_subject_name(probe_label),
                              "Success!" if limited_subjects_list[0][0] == probe_label else "Failure!"))
            return limited_subjects_list


        else:
            if probe_label is not None:
                print("Predicted class = {0} ({1}) with confidence = {2}; Actual class = {3} ({4}).\n\t Outcome: {5}"
                      .format(coll.getMinLabel(), get_subject_name(coll.getMinLabel()), coll.getMinDist(),
                              probe_label, get_subject_name(probe_label),
                              "Success!" if coll.getMinLabel() == probe_label else "Failure!"))
            return results



def test(model: cv.face_BasicFaceRecognizer, subj_list, image):
    mod, hei = train_recongizer(model, "../dataset_info/subjects.csv", resize=True)
    predict(subj_list, model=mod, height=hei, resize=True, probe_image=image, probe_label=6,
            show_mean=False, show_faces=False, identification=True)
    #predict(model=mod, height=hei, resize=True, probe_image="../images/dataset/cropped/s2/10.jpg", probe_label=2,
            #show_mean=False, show_faces=False, identification=False)
    #predict(model=mod, height=hei, resize=True, probe_image="../images/dataset/cropped/s8/22.jpg", probe_label=8,
            #show_mean=False, show_faces=False, identification=False)





if __name__ == '__main__':

    args = parse_args()

    out_dir = args.output
    image = args.input_image


    dir, file = path.split(image)
    dir_name = path.basename(dir)+'/'
    file_name, file_extension = path.splitext(file)

    save_dir = path.join(out_dir, dir_name)
    #print(save_dir)


    eyes_sf = args.eyes_scalefactor
    eyes_n = args.eyes_minneighbors
    eyes_ms = (args.eyes_minsize, args.eyes_minsize)

    #print('Chiamata a detect_cat_face')
    subj_list = detect_cat_eyes(image, show=True, eyes_ScaleFactor=eyes_sf, eyes_minNeighbors=eyes_n, eyes_minSize=eyes_ms)

    #print(len(subj_list))

    # Choose Recognizer
    if args.recognizer == 0:
        model: cv.face_BasicFaceRecognizer = cv.face.EigenFaceRecognizer_create()

    elif args.recognizer == 1:
        model: cv.face_BasicFaceRecognizer = cv.face.FisherFaceRecognizer_create()

    elif args.recognizer == 2:
        model: cv.face_BasicFaceRecognizer = cv.face.LBPHFaceRecognizer_create()


    if subj_list is None:
        new_subj_list = []
        for i in range(23):
            new_subj_list.append(str(i))
        subj_list = new_subj_list
    #print(len(subj_list))

    test(model, subj_list, image)





    


