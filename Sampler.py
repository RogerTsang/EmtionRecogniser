import cv2
import numpy as np
import datetime
from math import ceil, floor
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
        overlap = []

        is_overlap = False
        for i, (x, y, w, h) in enumerate(detected_faces):
            # Detect Face Overlap
            overlap.append((x, y, w, h))
            for (ox, oy, ow, oh) in overlap:
                if ox < x < ox + ow and oy < y < oy + oh:
                    is_overlap = True
                elif ox < x + w < ox + ow and oy < y < oy + oh:
                    is_overlap = True
                elif ox < x < ox + ow and oy < y + h + oy + oh:
                    is_overlap = True
                elif ox < x + w < ox + ow and oy < y + h < oy + oh:
                    is_overlap = True

            if is_overlap:
                logger.debug("Faces Overlap")
                continue

            # Outline Face
            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract Upper Face
            gray_upper_face = gray_frame[y:y + h/2, x:x + w]

            # Extract Eyes with estimation / haar-cascade
            detected_eyes = self._eye_cascade.detectMultiScale(gray_upper_face, 1.25, 3)
            if len(detected_eyes) != 2:
                eyes_minx, eyes_miny, eyes_maxx, eyes_maxy = self.estimate(w, h, "eyes")
                eyes_color = (75, 75, 150)
            else:
                minx = min(detected_eyes[0][0], detected_eyes[1][0]) - EYES_OFFSET_X
                miny = min(detected_eyes[0][1], detected_eyes[1][1]) - EYES_OFFSET_Y
                maxx = max(detected_eyes[0][0] + detected_eyes[0][2], detected_eyes[1][0] + detected_eyes[1][2]) + EYES_OFFSET_X
                maxy = max(detected_eyes[0][1] + detected_eyes[0][3], detected_eyes[1][1] + detected_eyes[1][3]) + EYES_OFFSET_Y
                # Calibration
                eyes_minx = max(minx, 0)
                eyes_miny = max(miny, 0)
                eyes_maxx = min(maxx, w)
                eyes_maxy = min(maxy, h / 2)
                eyes_color = (0, 0, 255)
            # Check if it needs to re-calibrate (reduce noise)
            eyes_length = eyes_maxx - eyes_minx
            eyes_height = eyes_maxy - eyes_miny
            if eyes_length < EYES_THRESHOLD_LENGTH * w or eyes_height < EYES_THRESHOLD_HEIGHT * h / 2:
                eyes_minx, eyes_miny, eyes_maxx, eyes_maxy = self.estimate(w, h, "eyes")
                eyes_color = (50, 50, 180)
            # Draw Eyes
            gray_eyes = gray_upper_face[eyes_miny: eyes_maxy, eyes_minx: eyes_maxx]
            cv2.rectangle(color_frame, (x + eyes_minx, y + eyes_miny), (x + eyes_maxx, y + eyes_maxy), eyes_color, 2)

            # Extract Lower Face
            gray_lower_face = gray_frame[y + h/2:y + h, x:x + w]
            # Extract Mouth with estimation / haar-cascade
            detected_mouths = self._mouth_cascade.detectMultiScale(gray_lower_face, 1.25, 3)
            if len(detected_mouths) != 1:
                mouth_minx, mouth_miny, mouth_maxx, mouth_maxy = self.estimate(w, h, "mouth")
                mouth_color = (150, 75, 75)
            else:
                mouth_minx = detected_mouths[0][0]
                mouth_miny = detected_mouths[0][1]
                mouth_maxx = detected_mouths[0][0] + detected_mouths[0][2]
                mouth_maxy = detected_mouths[0][1] + detected_mouths[0][3]
                mouth_color = (255, 0, 0)
            # Check if it needs to re-calibrate (reduce noise)
            mouth_length = mouth_maxx - mouth_minx
            mouth_height = mouth_maxy - mouth_miny
            if mouth_length < MOUTH_THRESHOLD_LENGTH * 2 or mouth_height < MOUTH_THRESHOLD_HEIGHT * h / 2:
                mouth_minx, mouth_miny, mouth_maxx, mouth_maxy = self.estimate(w, h, "mouth")
                mouth_color = (180, 50, 50)
            # Draw Mouth
            gray_mouth = gray_lower_face[mouth_miny: mouth_maxy, mouth_minx: mouth_maxx]
            cv2.rectangle(color_frame, (x + mouth_minx, y + h/2 + mouth_miny), (x + mouth_maxx, y + h/2 + mouth_maxy), mouth_color, 2)

            gray_mouth_resized = cv2.resize(gray_mouth, (OUTPUT_IMAGE_SIZE_X, OUTPUT_IMAGE_SIZE_Y))
            gray_eyes_resized = cv2.resize(gray_eyes, (OUTPUT_IMAGE_SIZE_X, OUTPUT_IMAGE_SIZE_Y))
            gray_resize = np.concatenate((gray_eyes_resized, gray_mouth_resized), axis=0)
            result.append((cv2.equalizeHist(gray_resize), x, y))
            
        return result

    def estimate(self, width, height, type):
        if type == "eyes":
            left = EYES_ESTIMATE_LEFT_MARGIN
            right = EYES_ESTIMATE_RIGHT_MARGIN
            top = EYES_ESTIMATE_TOP_MARGIN
            bottom = EYES_ESTIMATE_BOT_MARGIN
        elif type == "mouth":
            left = MOUTH_ESTIMATE_LEFT_MARGIN
            right = MOUTH_ESTIMATE_RIGHT_MARGIN
            top = MOUTH_ESTIMATE_TOP_MARGIN
            bottom = MOUTH_ESTIMATE_BOT_MARGIN
        else:
            logger.fatal("Error type in estimate()")

        minx = int(ceil(width * left))
        miny = int(ceil(height / 2 * top))
        maxx = int(floor(width * right))
        maxy = int(floor(height / 2 * bottom))
        return minx, miny, maxx, maxy

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
