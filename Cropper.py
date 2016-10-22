import os
import glob
import cv2

# Cropping is the first stage of the program
# It takes in the 'CKPath_Images' (Cohn-Kanade Database Root)
# Read all the face samples then crop them out with haarcascade_frontalface classifier
# Then store the cropped image to 'CKPath_Cropped' folder (preserve original folder structure)

OUTPUT_IMAGE_SIZE = 256
CKPath_Images = "C:\Users\zengh\Documents\CVision\Images"
CKPath_Cropped = "C:\Users\zengh\Documents\CVision\Cropped"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def main():
    # Traverse the Directory According To Different People's Faces
    images_group = glob.glob(CKPath_Images + "\\*")
    for imageGroup in images_group:
        image_id = imageGroup[-3:]
        image_dir = os.path.dirname(imageGroup)

        # Traverse Next Level By Emotion Categories
        motions_group = glob.glob(image_dir + "\\S" + image_id + "\\*")
        for motionGroup in motions_group:
            print motionGroup
            motion_id = motionGroup[-3:]
            motion_dir = os.path.dirname(motionGroup)

            # Traverse Each Images In the Emotion Directory
            faces_group = glob.glob(motion_dir + "\\" + motion_id + "\\*")
            for facesGroup in faces_group:
                # Read In Face Image
                face_basename = os.path.basename(facesGroup)
                face_image = cv2.imread(facesGroup, cv2.IMREAD_GRAYSCALE)

                # Create Directory For Cropped Image
                cropped_dir = CKPath_Cropped + "\\S" + image_id + "\\" + motion_id + "\\"
                if not os.path.exists(cropped_dir):
                    os.makedirs(cropped_dir)

                # Apply Haarcascade Classifier on Faces
                face_classified = face_cascade.detectMultiScale(face_image, 1.25, 5)
                for (x, y, w, h) in face_classified:
                    # Crop Faces Out From Images
                    cropped_image = face_image[y:y+h, x:x+h]

                    # Output Image needs to be the same size for FisherFace Recogniser
                    output_image = cv2.resize(cropped_image, (OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE))
                    cv2.imwrite(cropped_dir + face_basename, output_image)

if __name__ == '__main__':
    main()