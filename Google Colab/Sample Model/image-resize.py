import cv2
import glob
import os

inputFolder='/content/drive/My Drive/Dataset/test_set/right'
folderLen=len(inputFolder)
os.mkdir('right_test')
for img in glob.glob(inputFolder+"/*jpg"):
	image = cv2.imread(img)
	imgResized=cv2.resize(image,(256,256))
	cv2.imwrite("right_test"+img[folderLen:],imgResized)
