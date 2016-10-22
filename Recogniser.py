import cv2
import sys
import numpy as np
import time 
import logging
logging.basicConfig(level=logging.DEBUG)

from Trainer import *
from config import *
from Sampler import *

# Recognising is the third stage of the program
# It creates a Fisherface recogniser
# Trains it with the classified emotions from the data set
# Then predict the current emotion in the video

class Recognizer(object):

    def __init__(self, threshold=THRESHOLD, retrain_flag=True, recogniser='Fisher'):
        self._emotion_dictionary = emotions
        self._face_cascade = cv2.CascadeClassifier(CASCADE_FACE)
        self._mouth_cascade = cv2.CascadeClassifier(CASCADE_MOUTH)
        self._threshold = threshold
        self._retrain_flag = retrain_flag
        self._trainer = Trainer(recogniser)
        self._sampler = Sampler()

    def __call__(self):
        # Prediction on Video
        not_detected = cv2.imread("Not_Detected.png")
        recogniser = self._trainer.get_recognizer(self._retrain_flag)
        capture = cv2.VideoCapture(0)
        start = time.time()
        time.clock()
        elapsed = 0
        disp_emotion = []
        cache = list(dict())
        cache_emop = []

        logging.info(cache)

        while True:
            ret, color_frame = capture.read()
            results = self._sampler.extract(color_frame)
            elapsed = time.time() - start
            second = elapsed % 60

            for i, (gray_resize, x, y) in enumerate(results):
                prediction, confidence = recogniser.predict(gray_resize)
                emotion = self._emotion_dictionary[prediction]
                
                while len(disp_emotion) <= i:
                    disp_emotion.append('neutral')

                while len(cache) <= i:
                    cache.append({'neutral':0})

                if second >= 1:
                    if i == 0:
                        cache_emop = []

                    start = time.time()
                        
                    max_count = 0
                    logging.info('cache has %s', cache)
                    for _emotion, _count in cache[i].items():
                        if _count > max_count:
                            disp_emotion[i] = _emotion

                    cv2.putText(color_frame, str(disp_emotion[i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                    cache_emop.append((x,y))
                    # print "pic %s analysis Emotion: %10s | Confidence: %10f" % (i, disp_emotion[i], confidence)
                    
                    if i == len(results) - 1:
                        cache = list(dict())
                        # logging.info('cleared cache %s', cache)
                else:
                    cv2.putText(color_frame, str(disp_emotion[i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                    # print "pic %s analysis Emotion: %10s | Confidence: %10f" % (i, disp_emotion[i], confidence)
                    
                    if cache[i].get(emotion,None) is None:
                        cache[i][emotion] = 1
                    else:    
                        cache[i][emotion] += 1

            if len(results) == 0:
                logging.debug('emopo: %s', cache_emop)
                for i, emp in enumerate(cache_emop):
                    if i >= len(disp_emotion):
                        break
                    cv2.putText(color_frame, str(disp_emotion[i]), (emp[0], emp[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            cv2.imshow('face', color_frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                capture.release()
                cv2.destroyAllWindows()
                return 'peter cai'

if __name__ == '__main__':
    # rt_flag = False
    rt_flag = True
    if len(sys.argv) == 2:
        rt_flag = False if sys.argv[1] == '0' else True

    rec = Recognizer(recogniser='Eigen', retrain_flag=rt_flag)
    rec()