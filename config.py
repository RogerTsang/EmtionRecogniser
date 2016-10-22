import os
import logging
logging.basicConfig(level=logging.INFO)

# The emotion that we used for dispatching images from CKPath
emotions = {0:"neutral", 1:"anger", 2:"contempt", 3:"disgust", 4:"fear", 5:"happy", 6:"sadness", 7:"surprise", 8:"other"}
#{0:"neutral", 1:"anger", 2:"contempt", 3:"disgust", 4:"fear", 5:"happy", 6:"sadness", 7:"surprise", 8:"other"}

# The emotion table used for recognising live video
emotion_table = {"neutral": 0, "anger": 1, "happy": 5, "sadness": 6}
#{"neutral": 0, "anger": 1, "contempt": 2, "disgust": 3, "fear": 4, "happy": 5, "sadness": 6, "surprise": 7}

# The current working directory
CWD = os.path.dirname(__file__)

# The CK database root directory
CKPath_Root = "C:/Users/zengh/Documents/CVision"

# The folder for storing original images in CK database
CKPath_Images = "Images"

# The folder for storing dispatched emotions in CK database
CKPath_Dispatched = "Dispatched"

# The folder for storing emotion information in CK database
CKPath_Emotion = "Emotion"

# The threshold for recognising an emotion as other
THRESHOLD = 512

# The image size used for training
OUTPUT_IMAGE_SIZE = 256
OUTPUT_IMAGE_SIZE_X = 256
OUTPUT_IMAGE_SIZE_Y = 128

# Haar-Cascade description files used for
CASCADE_FACE = os.path.join(CKPath_Root, 'XML', 'haarcascade_frontalface_default.xml')
CASCADE_MOUTH = os.path.join(CKPath_Root, 'XML', 'haarcascade_mouth.xml')
CASCADE_EYE = os.path.join(CKPath_Root, 'XML', 'haarcascade_eye.xml')

# Training Cache
TRAINING_CACHE = "train.state"

if __name__ == '__main__':
    logging.info(CKPath_Root)
    logging.info(os.path.join(CKPath_Root,CKPath_Dispatched))
