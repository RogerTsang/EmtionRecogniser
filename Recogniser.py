import sys
import time
import operator
import logging
logging.basicConfig(level=logging.DEBUG)

from Trainer import *
from config import *
from Sampler import *

# Recognising is the third stage of the program
# It creates a recogniser
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
        recogniser = self._trainer.get_recognizer(self._retrain_flag)
        capture = cv2.VideoCapture(0)
        start = time.time()
        time.clock()
        disp_major_emotion = []
        disp_minor_emotion = []
        cache = list(dict())
        cache_emop = []

        logging.info(cache)

        while True:
            ret, color_frame = capture.read()
            results = self._sampler.extract(color_frame)

            for i, (gray_resize, x, y) in enumerate(results):
                prediction, confidence = recogniser.predict(gray_resize)
                emotion = self._emotion_dictionary[prediction]

                # Added multiple user record space for their emotion
                while len(disp_major_emotion) <= i:
                    disp_major_emotion.append('neutral')
                    disp_minor_emotion.append('')

                # Added multiple user emotion cache
                while len(cache) <= i:
                    cache.append({'neutral': 0})

                # Display emotion every second
                if time.time() - start >= 1:
                    if i == 0:
                        cache_emop = []

                    logging.info('cache has %s', cache)
                    # Fetch Emotion Ranking
                    sorted_emotion = sorted(cache[i].items(), key=operator.itemgetter(1), reverse=True)
                    var_emotion = np.var(cache[i].values())

                    # Check if the emotion can be recognised or not
                    if var_emotion >= THRESHOLD or var_emotion == 0:
                        disp_major_emotion[i] = sorted_emotion[0][0]
                    else:
                        disp_major_emotion[i] = '???'

                    # Check if the minor emotion should be the major guess
                    if var_emotion < THRESHOLD and var_emotion != 0:
                        disp_minor_emotion[i] = sorted_emotion[0][0] + '?'
                    elif len(sorted_emotion) > 1 and sorted_emotion[1][1] > 0:
                        disp_minor_emotion[i] = sorted_emotion[1][0] + '?'
                    else:
                        disp_minor_emotion[i] = ''

                    major_text_color = (0, 0, 255)
                    minor_text_color = (50, 50, 150)
                    text_font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(color_frame, str(disp_major_emotion[i]), (x, y), text_font, 1, major_text_color, 3)
                    cv2.putText(color_frame, str(disp_minor_emotion[i]), (x, y - 40), text_font, 0.8, minor_text_color, 2)

                    if i == len(results) - 1:
                        cache = list(dict())

                    start = time.time()
                else:
                    # print "pic %s analysis Emotion: %10s | Confidence: %10f" % (i, disp_emotion[i], confidence)
                    if cache[i].get(emotion, None) is None:
                        cache[i][emotion] = pow((16000 - confidence) // 1000, 2)
                    else:
                        cache[i][emotion] += pow((16000 - confidence) // 1000, 2)

            for i in range(len(results)):
                major_text_color = (0, 0, 255)
                minor_text_color = (50, 50, 150)
                text_font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(color_frame, str(disp_major_emotion[i]), (x, y), text_font, 1, major_text_color, 3)
                cv2.putText(color_frame, str(disp_minor_emotion[i]), (x, y - 40), text_font, 0.8, minor_text_color, 2)

            cv2.imshow('face', color_frame)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                capture.release()
                cv2.destroyAllWindows()
                return 0

if __name__ == '__main__':
    # rt_flag = False
    rt_flag = False
    if len(sys.argv) == 2:
        rt_flag = False if sys.argv[1] == '0' else True

    rec = Recognizer(recogniser='Eigen', retrain_flag=rt_flag)
    rec()
