# USAGE
# python annotate.py --input downloads --annot dataset

# import the necessary packages
from imutils import paths
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True,
	help="path to output directory of annotations")
args = vars(ap.parse_args())

# grab the image paths then initialize the dictionary of character
# counts
imagePaths = list(paths.list_images(args["input"]))
counts = {}

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# display an update to the user
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))

	try:
		# load the image and pad the
		# image to ensure digits caught only the border of the image
		# are retained
		image = cv2.imread(imagePath)
		image = cv2.copyMakeBorder(image, 10, 10, 10, 10,
			cv2.BORDER_REPLICATE)


		# display the character, making it larger enough for us
		# to see, then wait for a keypress
		cv2.imshow("Image", imutils.resize(image, width=28))
		key = cv2.waitKey(0)

		# if the '`' key is pressed, then ignore the character
		if key == ord("`"):
			print("[INFO] ignoring character")
			continue

		# grab the key that was pressed and construct the path
		# the output directory
		key = chr(key).upper()
		dirPath = os.path.sep.join([args["annot"], key])
		
		# if the output directory does not exist, create it
		if not os.path.exists(dirPath):
			os.makedirs(dirPath)

		# write the labeled character to file
		count = counts.get(key, 1)
		p = os.path.sep.join([dirPath, "{}.png".format(
			str(count).zfill(6))])
		cv2.imwrite(p, image)
	
		# increment the count for the current key
		counts[key] = count + 1

	# we are trying to control-c out of the script, so break from the
	# loop (you still need to press a key for the active window to
	# trigger this)
	except KeyboardInterrupt:
		print("[INFO] manually leaving script")
		break

	# an unknown error has occurred for this particular image
	except:
		print("[INFO] skipping image...")
