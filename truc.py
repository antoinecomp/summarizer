# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os

# libraries
## split
from PyPDF2 import PdfFileWriter, PdfFileReader
# remove
import sys
# 
from pdf2image import convert_from_path

def pdfspliter(filename):
	inputpdf = PdfFileReader(open(filename, "rb"))
	for i in range(inputpdf.numPages):
		output = PdfFileWriter()
		output.addPage(inputpdf.getPage(i))
		with open("document-page%s.pdf" % i, "wb") as outputStream:
			output.write(outputStream)
		pages = convert_from_path("document-page%s.pdf" % i, 500)
		for page in pages:
			page.save('out.jpg', 'JPEG')
		imageparser('out.jpg')
		os.remove("document-page%s.pdf" % i)

		
def imageparser(pdf):
	print("in imageparser")
	# argument parser
	#ap = argparse.ArgumentParser()
	#ap.add_argument("-i", "--image", required=False, # used to be True
	#	help="path to input image to be OCR'd")
	#ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	#	help="type of preprocessing to be done")
	#args = vars(ap.parse_args())
	print("in imageparser")
	# load the example image and convert it to grayscale
	# image = cv2.imread(args["image"])
	image = pdf
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	 
	# check to see if we should apply thresholding to preprocess the
	# image
	# if args["preprocess"] == "thresh": # used to be that
	if True :
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

	# load the image as a PIL/Pillow image, apply OCR, and then delete
	# the temporary file
	text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	print(text)

	# show the output images
	cv2.imshow("Image", image)
	cv2.imshow("Output", gray)
	cv2.waitKey(0)

if __name__ == '__main__':
	filename = sys.argv[1]
	pdfspliter(filename)
	
	
	
	
