import os
import cv2
import numpy as np
import datetime
from config import *

import logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.info('Start')

class Sampler(object):

    def __init__(self, path=os.path.join(CKPath_Root, CKPath_Dispatched)):
        self._path = path
        self._pathdic = emotions
        self._face_cascade = cv2.CascadeClassifier(CASCADE_FACE)
        self._mouth_cascade = cv2.CascadeClassifier(CASCADE_MOUTH)
        self._eye_cascade = cv2.CascadeClassifier(CASCADE_EYE)
        logger.info(emotions)

    def extract(self, color_frame):
        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self._face_cascade.detectMultiScale(gray_frame, 1.25, 5)
        # gray_frame = cv2.equalizeHist(gray_frame)
        result = []

        for i, (x, y, w, h) in enumerate(detected_faces):
            gray_face = gray_frame[y + h/2:y + h, x:x + w]
            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            detected_mouths = self._mouth_cascade.detectMultiScale(gray_face, 1.25, 3)
            if len(detected_mouths) != 1:
                continue

            (mx, my, mw, mh) = detected_mouths[0]
            gray_mouth = gray_face[my:my + mh, mx: mx + mw]
            cv2.rectangle(color_frame, (x + mx, y + my + h/2), (x + mx + mw, y + h/2 + my + mh), (255, 0, 0), 2)

            gray_face = gray_frame[y:y + h/2, x:x + w]
            detected_eyes = self._eye_cascade.detectMultiScale(gray_face, 1.25, 3)

            if len(detected_eyes) != 2:
                continue

            # logging.info(detected_eyes)
            minx = min(detected_eyes[0][0], detected_eyes[1][0]) - EYES_OFFSET_X
            miny = min(detected_eyes[0][1], detected_eyes[1][1]) - EYES_OFFSET_Y
            maxx = max(detected_eyes[0][0] + detected_eyes[0][2], detected_eyes[1][0] + detected_eyes[1][2]) + EYES_OFFSET_X
            maxy = max(detected_eyes[0][1] + detected_eyes[0][3], detected_eyes[1][1] + detected_eyes[1][3]) + EYES_OFFSET_Y

            # Calibration
            eyes_minx = max(minx, 0)
            eyes_miny = max(miny, 0)
            eyes_maxx = min(maxx, w)
            eyes_maxy = min(maxy, h/2)
            gray_eyes = gray_face[eyes_miny:eyes_maxy, eyes_minx:eyes_maxx]
            cv2.rectangle(color_frame, (x + eyes_minx, y + eyes_miny), (x + eyes_maxx, y + eyes_maxy), (0, 0, 255), 2)

            gray_mouth_resized = cv2.resize(gray_mouth, (OUTPUT_IMAGE_SIZE_X, OUTPUT_IMAGE_SIZE_Y))
            gray_eyes_resized = cv2.resize(gray_eyes, (OUTPUT_IMAGE_SIZE_X, OUTPUT_IMAGE_SIZE_Y))
            gray_resize = np.concatenate((gray_eyes_resized, gray_mouth_resized), axis=0)
            result.append((cv2.equalizeHist(gray_resize), x, y))
            
            # cv2.imshow('result', gray_resize)

        return result

    def __call__(self):
        capture = cv2.VideoCapture(0)

        while True:
            ret, color_frame = capture.read()
            result = self.extract(color_frame)

            if len(result) == 0:
                continue

            grey_resize, _, _ = result[0]
            cv2.imshow('face', color_frame)
            
            k = cv2.waitKey(30) & 0xff
            if 56 >= k >= 48:
                action = k - 48
                emotion = self._pathdic.get(action, None)
                if emotion is None:
                    logging.info('Invalid Emotion index %s' % action)
                    continue

                timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
                picname = '%s_%s.png' % (emotion, timestamp)
                print picname
                filename = os.path.join(self._path, emotion, picname)
                print(filename)
                
                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))
                
                cv2.imwrite(filename, grey_resize)
                cv2.imshow('screenshot', grey_resize)    
                continue
            elif k == 27:
                capture.release()
                cv2.destroyAllWindows()
                return 0
            else:
                continue

if __name__ == '__main__':
    sam = Sampler()
    sam.__call__()
