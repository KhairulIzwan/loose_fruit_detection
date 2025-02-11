##################################################
## {Description}: Rename and resize the dataset by 
## numbering format e.g.: 000001.png
##################################################
## Author: Nurshafikah Binti Darwis
## Version: {1}.{0}.{1}
## Email: {shafikahdarwis@gmail.com}
##################################################

# import the necessary packages
from imutils import paths
import argparse
import imutils
import cv2
import os

from dl4cv_pyimagesearch_preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True,
	help="path to output directory of annotations")
ap.add_argument("-f", "--folder", required=True, 
	help = "foldername")
ap.add_argument("-ws", "--width", required=True,
	help="width size")
ap.add_argument("-hs", "--height", required=True,
	help="height size")
args = vars(ap.parse_args())

# grab the image paths then initialize the dictionary of character
# counts
imagePaths = list(paths.list_images(args["input"]))
count = 0

# initiate aspectawarepreprocessor
aap = AspectAwarePreprocessor(int(args["width"]), int(args["height"]))

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# display an update to the user
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))

	try:
		# load the image
		image = cv2.imread(imagePath)

		# resize the image
		image = aap.preprocess(image)

		# construct the path the output directory
		dirPath = os.path.join(args["annot"], args["folder"])

		# if the output directory does not exist, create it
		if not os.path.exists(dirPath):
			os.makedirs(dirPath)

		# write the labeled character to file
		p = os.path.sep.join([dirPath, "{}_W{}H{}.png".format(
			str(count).zfill(6), args["width"], args["height"])])
		cv2.imwrite(p, image)

		# increment the count for the current key
		count = count + 1

	# we are trying to control-c out of the script, so break from the
	# loop (you still need to press a key for the active window to
	# trigger this)
	except KeyboardInterrupt:
		print("[INFO] manually leaving script")
		break

	# an unknown error has occurred for this particular image
	except:
		print("[INFO] skipping image...")
