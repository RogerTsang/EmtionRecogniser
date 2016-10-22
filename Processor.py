import glob
import cv2
import numpy as np

from config import *


class Processor(object):
    def __init__(self):
        self._path_images = os.path.join(CKPath_Root, CKPath_Images)
        self._path_emotions = os.path.join(CKPath_Root, CKPath_Emotion)
        self._path_dispatched = os.path.join(CKPath_Root, CKPath_Dispatched)
        self._emotion_dictionary = emotions
        self._face_cascade = cv2.CascadeClassifier(CASCADE_FACE)
        self._eye_cascade = cv2.CascadeClassifier(CASCADE_EYE)
        self._mouth_cascade = cv2.CascadeClassifier(CASCADE_MOUTH)

    def process(self):
        original_group = glob.glob(os.path.join(self._path_images, "*"))
        original_len = len(original_group)
        for dir_index, (original_dir) in enumerate(original_group):
            # Fetch candidate group from file directory
            is_neutral_processed = False
            candidate = os.path.basename(original_dir)
            print "Processing candidate : " + candidate + " Index: " + str(dir_index + 1) + " / " + str(original_len)

            # Fetch emotion encoding for each group
            emotion_group = glob.glob(os.path.join(original_dir, "*"))
            for emotion_dir in emotion_group:
                # Fetch emotion directory from each candidate
                emotion = os.path.basename(emotion_dir)

                # Fetch emotion encoding from CKPath_emotion directories
                encoding_group = glob.glob(os.path.join(self._path_emotions, candidate, emotion, "*"))
                emotion_name = None
                for encoding_dir in encoding_group:
                    # Open emotion file
                    emotion_file = open(encoding_dir, "r")
                    emotion_encoding = int(float(emotion_file.readline()))
                    emotion_file.close()

                    # Search for emotion name from config
                    emotion_name = self._emotion_dictionary[emotion_encoding]

                # If the CK image belongs to one of the emotion set
                # Copy the last image from the directory (the peak emotion) into our own database
                if emotion_name is None:
                    continue

                # Fetch images from directory
                image_dir = glob.glob(os.path.join(emotion_dir, "*"))
                image_name = os.path.basename(image_dir[0])

                if not is_neutral_processed:
                    # Process and combine neutral emotion
                    neutral_image = cv2.imread(image_dir[0], cv2.IMREAD_GRAYSCALE)
                    neutral_combine = self.combine(neutral_image)

                    # Store processed neutral emotion
                    neutral_dir = os.path.join(self._path_dispatched, "neutral")
                    if neutral_combine is not None:
                        if not os.path.exists(neutral_dir):
                            os.makedirs(neutral_dir)
                        cv2.imwrite(os.path.join(neutral_dir, image_name), neutral_combine)

                    # Set neutral as processed
                    is_neutral_processed = True

                # Process and combine neutral emotion
                peak_image = cv2.imread(image_dir[-1], cv2.IMREAD_GRAYSCALE)
                peak_combine = self.combine(peak_image)

                # Store processed emotional image
                peak_dir = os.path.join(self._path_dispatched, emotion_name)
                if peak_combine is not None:
                    if not os.path.exists(peak_dir):
                        os.makedirs(peak_dir)
                    cv2.imwrite(os.path.join(peak_dir, image_name), peak_combine)

    def combine(self, image):
        # Apply Haar-cascade to neutral image
        cascade_face = self._face_cascade.detectMultiScale(image, 1.25, 5)
        if len(cascade_face) != 1:
            return None

        (image_x, image_y, image_w, image_h) = cascade_face[0]
        image_upper = image[image_y: image_y + image_h / 2, image_x:image_x + image_w]
        image_lower = image[image_y + image_h / 2: image_y + image_h, image_x:image_x + image_w]

        # Cascade Eye
        cascade_eyes = self._eye_cascade.detectMultiScale(image_upper, 1.25, 3)
        # If the number of eyes if not equal to 2, abort
        if len(cascade_eyes) != 2:
            return None

        # Crop out both eyes
        eyes_minx = min(cascade_eyes[0][0], cascade_eyes[1][0]) - EYES_OFFSET_X
        eyes_miny = min(cascade_eyes[0][1], cascade_eyes[1][1]) - EYES_OFFSET_Y
        eyes_maxx = max(cascade_eyes[0][0] + cascade_eyes[0][2], cascade_eyes[1][0] + cascade_eyes[1][2]) + EYES_OFFSET_X
        eyes_maxy = max(cascade_eyes[0][1] + cascade_eyes[0][3], cascade_eyes[1][1] + cascade_eyes[1][3]) + EYES_OFFSET_Y
        image_eyes = image_upper[eyes_miny:eyes_maxy, eyes_minx:eyes_maxx]

        # Cascade Mouth
        cascade_mouth = self._mouth_cascade.detectMultiScale(image_lower, 1.25, 3)
        # If the number of mouths if not equal to 1, abort
        if len(cascade_mouth) == 1:
            # Crop out mouth
            mouth_minx = cascade_mouth[0][0]
            mouth_miny = cascade_mouth[0][1]
            mouth_maxx = cascade_mouth[0][0] + cascade_mouth[0][2]
            mouth_maxy = cascade_mouth[0][1] + cascade_mouth[0][3]
        # If the mouth cannot be found, crop it out manually according to the eyes position
        else:
            mouth_center_x = (eyes_minx + eyes_maxx) / 2
            mouth_center_y = image_h / 4
            mouth_minx = mouth_center_x - 50
            mouth_miny = mouth_center_y - 20
            mouth_maxx = mouth_center_x + 70
            mouth_maxy = mouth_center_y + 20

        image_mouth = image_lower[mouth_miny:mouth_maxy, mouth_minx:mouth_maxx]

        # Combine them together
        image_eyes_resize = cv2.resize(image_eyes, (OUTPUT_IMAGE_SIZE_X, OUTPUT_IMAGE_SIZE_Y))
        image_mouth_resize = cv2.resize(image_mouth, (OUTPUT_IMAGE_SIZE_X, OUTPUT_IMAGE_SIZE_Y))
        image_combined = np.concatenate((image_eyes_resize, image_mouth_resize), axis=0)

        # Apply Histogram Average
        image_average = cv2.equalizeHist(image_combined)
        return image_average

if __name__ == '__main__':
    proc = Processor()
    proc.process()