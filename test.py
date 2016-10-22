import os
import glob
import cv2
import numpy as np
import Sampler

path = os.path.join(os.path.dirname(__file__), 'Dispatch_bak', '*')
save = os.path.join(os.path.dirname(__file__), 'Dispatch')
sam = Sampler.Sampler()

for folders in glob.glob(path):
	images_dir = os.path.join(folders, '*')
	for files in glob.glob(images_dir):
		faces = cv2.imread(files)
		result = sam.extract(faces)
		if len(result) != 0:
			path_save = os.path.join(save,os.path.basename(folders),os.path.basename(files))
			cv2.imwrite(path_save, np.asarray(result[0]))
