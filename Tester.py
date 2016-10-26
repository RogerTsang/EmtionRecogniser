#!/usr/bin/python
import glob
import cv2
import os
import numpy as np
import time

from config import *
from Sampler import Sampler
from Trainer import Trainer

class Tester(object):
    def __init__(self, s, t):
        self._sampler = s
        self._emotion_confused = emotion_confused
        self._emotion_dictionary = emotions
        self._recogniser = t.get_recognizer(False)

    def eval(self, image, correct_emotion):
        results = self._sampler.extract(image)
        emotion = None
        correct = None
        for i, (gray_resize, x, y) in enumerate(results):
            prediction, confidence = self._recogniser.predict(gray_resize)
            emotion = self._emotion_dictionary[prediction]
            if(emotion == correct_emotion):
                correct = True
            else:
                correct = False

        return (emotion, correct)

    def static_test(self, filepath=os.path.join(CKPath_Root, CKPath_Benchmark)):
        benchmark_flies = glob.glob(os.path.join(filepath, '*'))
        predict_true = 0
        predict_false = 0
        not_reco = 0

        confusion_matrix = list(list())
        for i in range(5):
            confusion_matrix.append([0] * 5)

        for file in benchmark_flies:
            emotion = os.path.basename(file)
            for images in glob.glob(os.path.join(file, '*')):
                image = cv2.imread(images)
                pred_emotion, res = self.eval(image, emotion)

                if res == True:
                    predict_true += 1
                elif res == False:
                    predict_false += 1
                else:
                    not_reco += 1
                    continue

                confusion_matrix[self._emotion_confused[emotion]][self._emotion_confused[pred_emotion]] += 1

        if predict_true + predict_false > 0:
            print("Rate of success: ", predict_true * 1.0/(predict_true + predict_false))
        print("%s miss" % not_reco)
        print(confusion_matrix)

    def live_test(self):
        capture = cv2.VideoCapture(0)
        confusion_matrix = list(list())

        cache = dict({'neutral':0})
        not_reco = 0
        predict_true = 0
        predict_false = 0

        for i in range(5):
            confusion_matrix.append([0] * 5)

        start = time.time()
        time.clock()

        while True:
            ret, color_frame = capture.read()
            results = self._sampler.extract(color_frame)

            for i, (gray_resize, x, y) in enumerate(results):
                prediction, confidence = self._recogniser.predict(gray_resize)
                emotion = self._emotion_dictionary[prediction]

                if time.time() - start >= 1:
                    cache = dict({'neutral':0})
                    start = time.time()
                else:
                    if cache.get(emotion, None) is None:
                        cache[emotion] = 1
                    else:
                        cache[emotion] += 1


            cv2.imshow('face', color_frame)
            k = cv2.waitKey(30) & 0xff
            if 56 >= k >= 48:
                correct_emotion = k - 48
                pred_emotion = 'neutral'
                count = 0

                for key, item in cache.items():
                    if item > count:
                        maxim = key
                        count = item

                if count == 0:
                    not_reco += 1
                    continue

                if correct_emotion == pred_emotion:
                    predict_true += 1
                else:
                    predict_false += 1

                confusion_matrix[self._emotion_confused[emotion]][self._emotion_confused[pred_emotion]] += 1

            elif k == 27:
                capture.release()
                cv2.destroyAllWindows()
                if predict_true + predict_false <= 0:
                    print("%s miss" % not_reco)
                    return

                print("Rate of success: ", predict_true * 1.0/(predict_true + predict_false))
                print("%s miss" % not_reco)
                print(confusion_matrix)
            

if __name__ == '__main__':
    # s = Sampler()
    # t = Trainer('Eigen')
    # tes = Tester(s,t)
    # tes.static_test()