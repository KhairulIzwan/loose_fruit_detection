# USAGE
# python augmentation_demo.py --image jemma.png --output output

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
from imutils import paths
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store augmentation examples")
ap.add_argument("-p", "--prefix", type=str, default="image",
	help="output filename prefix")
args = vars(ap.parse_args())

# grab the image paths then initialize the dictionary of character
# counts
imagePaths = list(paths.list_images(args["image"]))

# construct the path the output directory
dirPath = os.path.join(args["output"], args["prefix"])

# if the output directory does not exist, create it
if not os.path.exists(dirPath):
	os.makedirs(dirPath)
	
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# load the input image, convert it to a NumPy array, and then
	# reshape it to have an extra dimension
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
	image = load_img(imagePath)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# construct the image generator for data augmentation then
	# initialize the total number of images generated thus far
	aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
		height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
		horizontal_flip=True, fill_mode="nearest")
	total = 0

	# construct the actual Python generator
	print("[INFO] generating images...")
	imageGen = aug.flow(image, batch_size=1, save_to_dir=dirPath,
		save_prefix=args["prefix"], save_format="jpg")

	# loop over examples from our image data augmentation generator
	for image in imageGen:
		# increment our counter
		total += 1

		# if we have reached 10 examples, break from the loop
		if total == 1000:
			break
