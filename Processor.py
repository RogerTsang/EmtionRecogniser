import os
import glob
import cv2

from config import *

class Processor(object):
    def __init__(self):
        self._path_images = os.path.join(CKPath_Root, CKPath_Images)
        self._path_emotions = os.path.join(CKPath_Root, CKPath_Emotion)
        self._eye_cascade = CASCADE_EYE
        self._mouth_cascade = CASCADE_MOUTH

    def process(self):
        original_group = glob.glob(os.path.join(self._path_images, "*"))
        for original_dir in original_group:
            # Fetch candidate group from file directory
            candidate = os.path.basename(original_dir)

            

if __name__ == '__main__':
    proc = Processor()
    proc.process()