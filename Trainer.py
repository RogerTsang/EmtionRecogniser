#!/usr/bin/python
import cv2
import glob
import numpy as np
import os

import logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info('Start')

import config
from config import CKPath_Dispatched, CKPath_Root
emotion_table = config.emotion_table

class Trainer(object):
    def __init__(self,recogniser):
        self._recogniser = None
        self._recname = recogniser
        
        (major, minor, _) = cv2.__version__.split(".")
        if recogniser == 'Fisher':
            if major == '3':
                self._recogniser = cv2.face.createFisherFaceRecognizer()
            else:
               self._recogniser = cv2.createFisherFaceRecognizer() 
        elif recogniser == 'Eigen':
            if major == '3':
                self._recogniser = cv2.face.createEigenFaceRecognizer()
            else:
               self._recogniser = cv2.createEigenFaceRecognizer() 
            
    def get_recognizer(self,retrain_flag = True):

        # Check If Trainer Has Cached Previous Result on the File System
        state_file = os.path.join(CKPath_Root, TRAINING_CACHE)
        if not os.path.isfile(state_file) or retrain_flag:
            is_trained = False
            self.train(emotion_table, is_trained)
            self._recogniser.save(state_file)

        else:
            print "No Training Required. Load From File"
            self._recogniser.load(state_file)

        return self._recogniser


    def train(self,emotion_table, is_trained=False):

        # Initialise FisherFace recognizer and training pair
        global recogniser

        images = []
        labels = []

        print "Training on demand"

        # Fetch Emotion Images From Desired Folders
        for emotion_group in emotion_table.viewkeys():
            file_path = os.path.join(CKPath_Root, os.path.join(CKPath_Dispatched,os.path.join(emotion_group,'*')))
            logging.debug(file_path)

            file_group = glob.glob(file_path)
            logger.info(emotion_group)

            for emotion_file in file_group:
                input_image = cv2.imread(emotion_file, cv2.IMREAD_GRAYSCALE)
                input_image = cv2.equalizeHist(input_image)
                images.append(input_image)
                labels.append(emotion_table[emotion_group])

        # Train the Recogniser
        labels = np.asarray(labels)
        self._recogniser.train(images, labels)

        # Save Recogniser Model State

if __name__ == '__main__':
    get_recognizer()
