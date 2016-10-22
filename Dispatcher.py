import os
import glob
import shutil

# Dispatching is the second stage of the program
# It takes in the 'CKPath_Emotion' which stores the encoding of emotion of each set of faces
# According to Cohn-Kanade documentation, the emotion encoding are listed below
# 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
# We dispatch each kind of emotion to 'CKPath_Dispatched' directory

CKPath_Cropped = "C:\Users\zengh\Documents\CVision\Cropped"
CKPath_Emotion = "C:\Users\zengh\Documents\CVision\Emotion"
CKPath_Dispatched = "C:\Users\zengh\Documents\CVision\Dispatched"

Emotion_Dictionary = {0:"neutral", 1:"anger", 2:"contempt", 3:"disgust", 4:"fear", 5:"happy", 6:"sadness", 7:"surprise"}

PEAK_THRESHOLD = 1 # Number of peak emotions are copied to dispatched directory
NEUTRAL_THRESHOLD = 1 # Number of neutral emotions are copied to dispatched directory


def main():
    # Create Neutral Directory
    netural_dir = CKPath_Dispatched + "\\" + "neutral" + "\\"
    if not os.path.exists(netural_dir):
        os.makedirs(netural_dir)

    # Traverse the Directory According To Different People's Faces
    images_group = glob.glob(CKPath_Emotion + "\\*")
    for imageGroup in images_group:
        image_id = imageGroup[-3:]
        image_dir = os.path.dirname(imageGroup)

        # Traverse Next Level By Emotion Categories
        motions_group = glob.glob(image_dir + "\\S" + image_id + "\\*")
        for motionGroup in motions_group:
            motion_id = motionGroup[-3:]
            motion_dir = os.path.dirname(motionGroup)

            # Traverse Each Images In the Emotion Directory
            encoding_group = glob.glob(motion_dir + "\\" + motion_id + "\\*")
            for encodingGroup in encoding_group:
                # Read In Face Emotion Encoding File
                emotion_file = open(encodingGroup, "r")

                # Fetch the Emotion Encoding From the file
                emotion_encoding = int(float(emotion_file.read()))
                emotion_name = Emotion_Dictionary.get(emotion_encoding)

                if emotion_name is not None:
                    dispatched_dir = CKPath_Dispatched + "\\" + emotion_name + "\\"

                    # Copy the Images from Cropped Directories to Dispatched
                    cropped_dir = CKPath_Cropped + "\\S" + image_id + "\\" + motion_id + "\\"
                    cropped_group = glob.glob(cropped_dir + "*")

                    # Create OS Path for Emotion
                    if not os.path.exists(dispatched_dir):
                        os.makedirs(dispatched_dir)

                    # Picked the Last 3 Peak Image to Emotion Directory
                    for i in range(-PEAK_THRESHOLD, 0):
                        shutil.copy(cropped_group[i], dispatched_dir)
                        print cropped_group[i]

                    # Picked the First 2 Image to Neutral Directory
                    for j in range(0, NEUTRAL_THRESHOLD):
                        shutil.copy(cropped_group[j], netural_dir)

main()