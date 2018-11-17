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
# To import the dominant colors
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches




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

### Attempt of group algorithm
# we get the dominant colors
img = cv2.imread('strike.png')
height, width, dim = img.shape
# We take only the center of the image
img = img[int(height/4):int(3*height/4), int(width/4):int(3*width/4), :]
height, width, dim = img.shape

img_vec = np.reshape(img, [height * width, dim] )

kmeans = KMeans(n_clusters=3)
kmeans.fit( img_vec )

#  count cluster pixels, order clusters by cluster size
unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)
sort_ix = np.argsort(counts_l)
sort_ix = sort_ix[::-1]

fig = plt.figure()
ax = fig.add_subplot(111)
x_from = 0.05

for cluster_center in kmeans.cluster_centers_[sort_ix]:
    ax.add_patch(patches.Rectangle( (x_from, 0.05), 0.29, 0.9, alpha=None,
                                    facecolor='#%02x%02x%02x' % (int(cluster_center[2]), int(cluster_center[1]), int(cluster_center[0]) ) ) )
    x_from = x_from + 0.31

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
	
	# Here we should split the images in parts. Those who have strokes
	# We asked for a stroke example so we have its color 
	# While we find shape with the same color we store all the full line of pixels
	
	
	
	# Or we can group strokes and recognize text

	# load the image as a PIL/Pillow image, apply OCR, and then delete
	# the temporary file
	text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	#print(text)

	with open('resume.txt', 'a+') as f:
		print('***:', text, file=f)  
		





