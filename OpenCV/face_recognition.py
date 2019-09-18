# check opencv version
import cv2
# print version number
print(cv2.__version__)
# https://docs.opencv.org/4.1.1/db/d28/tutorial_cascade_classifier.html
# load the pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')