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

# For Tesseract to work on windows
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
		


# removes pixels in image that are between the range of
# [lower_val,upper_val]
def remove_gray(img,lower_val,upper_val):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0,0,lower_val])
    upper_bound = np.array([255,255,upper_val])
    mask = cv2.inRange(gray, lower_bound, upper_bound)
    return cv2.bitwise_and(gray, gray, mask = mask)
	
# erodes image based on given kernel size (erosion = expands black areas)
def erode( img, kern_size = 3 ):
    retval, img = cv2.threshold(img, 254.0, 255.0, cv2.THRESH_BINARY) # threshold to deal with only black and white.
    kern = np.ones((kern_size,kern_size),np.uint8) # make a kernel for erosion based on given kernel size.
    eroded = cv2.erode(img, kern, 1) # erode your image to blobbify black areas
    [y,x,z] = eroded.shape # get shape of image to make a white boarder around image of 1px, to avoid problems with find contours.
    return cv2.rectangle(eroded, (0,0), (x,y), (255,255,255), 1)
	
# finds contours of eroded image
def prep( img, kern_size = 3 ):    
    img = erode( img, kern_size )
    retval, img = cv2.threshold(img, 200.0, 255.0, cv2.THRESH_BINARY_INV) #   invert colors for findContours
    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # Find Contours of Image
	
# given img & number of desired blobs, returns contours of blobs.
def blobbify(img, num_of_labels, kern_size = 3, dilation_rate = 10):
    prep_img, contours, hierarchy = prep( img.copy(), kern_size ) # dilate img and check current contour count.
    while len(contours) > num_of_labels:
        kern_size += dilation_rate # add dilation_rate to kern_size to increase the blob. Remember kern_size must always be odd.
        previous = (prep_img, contours, hierarchy)
        processed_img, contours, hierarchy = prep( img.copy(), kern_size ) # dilate img and check current contour count, again.
    if len(contours) < num_of_labels:
        return (processed_img, contours, hierarchy)
    else:
        return previous

# finds bounding boxes of all contours
def bounding_box(contours):
    bBox = []
    for curve in contours:
        box = cv2.boundingRect(curve)
    bBox.append(box)
    return bBox
		
# variables 
start = -1
num_of_labels = [] #meta data for each image


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

for file_name in file_names:
	print("we wrote : ",file_name)
	# load the image and convert it to grayscale
	img = cv2.imread(file_name)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
	
	### Prijatelj
	img = erode( img, kern_size = 3 )
	contours = prep( img, kern_size = 3 )
	# blobbify(img, num_of_labels, kern_size = 3, dilation_rate = 10)
	print(bounding_box(contours))





