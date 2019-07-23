# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
## split
from PyPDF2 import PdfFileWriter, PdfFileReader
# remove
import sys
# 
from pdf2image import convert_from_path
# import all files with a name
import glob
# to import the dominant colors
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

# functions
def pdfspliterimager(filename):
	inputpdf = PdfFileReader(open(filename, "rb"))
	for i in range(inputpdf.numPages):
		output = PdfFileWriter()
		output.addPage(inputpdf.getPage(i))
		with open("document-page%s.pdf" % i, "wb") as outputStream:
			output.write(outputStream)
		pages = convert_from_path("document-page%s.pdf" % i, 500)
		for page in pages:
			page.save('out%s.jpg'%i, 'JPEG')
		
		os.remove("document-page%s.pdf" % i)

# variables 
start = -1

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())

# we test if it is a pdf
image_path = args["image"]
# if it is a pdf we convert it to an image
if image_path.endswith('.pdf'):
	pdfspliterimager(image_path)

# we store all images with out in their name
file_names = glob.glob("out*")
file_names= sorted(file_names)

### Attempt to get the colors of the stroke example
# we get the dominant colors
#img = cv2.imread('strike.png')
#height, width, dim = img.shape
# We take only the center of the image
#img = img[int(height/4):int(3*height/4), int(width/4):int(3*width/4), :]
#height, width, dim = img.shape

#img_vec = np.reshape(img, [height * width, dim] )

#kmeans = KMeans(n_clusters=2)
#kmeans.fit( img_vec )

#  count cluster pixels, order clusters by cluster size
#unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)
#sort_ix = np.argsort(counts_l)
#sort_ix = sort_ix[::-1]

#fig = plt.figure()
#ax = fig.add_subplot(111)
#x_from = 0.05

# what is this ?
#cluster_center = kmeans.cluster_centers_[sort_ix][1]

# plt.show()
### End of attempt

for file_name in file_names:
	print("we wrote : ",file_name)
	# load the image and convert it to grayscale
	image = cv2.imread(file_name)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# check to see if we should apply thresholding to preprocess the
	# image
	if args["preprocess"] == "thresh":
		gray = cv2.threshold(gray, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# make a check to see if median blurring should be done to remove
	# noise
	elif args["preprocess"] == "blur":
		gray = cv2.medianBlur(gray, 3)

	# write the grayscale image to disk as a temporary file so we can
	# apply OCR to it
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, gray)
	
	# Here we should split the images in two parts : those who has strokes and those who don t
	# We asked for a stroke example so we have its color 
	# While we find pixels with the same color we store its line
	im = Image.open(filename)
	(width, height)= im.size
	for x in range(width): 
		for y in range(height):
			rgb_im = im.convert('RGB')
			red, green, blue = rgb_im.getpixel((1, 1))
			# We test if the pixel has the same color as the second cluster # We should rather test if it is "alike"
			# It means that we found a line were there is some paper stroke
			if np.array_equal([red,green,blue],cluster_center): 
				# if it is the case we store the width as starting point while we find pixels 
				# and we break the loop to go to another line
				if start == -1:
					start = x
					selecting_area = True
					break
				# if it already started we break the loop to go to another line
				if selecting_area == True:
					break
			# if no pixel in a line had the same color as the second cluster but selecting already started
			# we crop the image and go to another line
			# it means that there is no more paper stroke
			if selecting_area == True:
				text_box = (0, start, width, x)
				# Crop Image
				area = im.crop(text_box)
				area.show()	
				break
		    
	
	
	# Or we can group strokes and recognize text

	# load the image as a PIL/Pillow image, apply OCR, and then delete
	# the temporary file
	text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	#print(text)

	with open('resume.txt', 'a+') as f:
		print('***:', text, file=f)  
		





