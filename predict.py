# Proof-of-concept
import cv2
import sys
from models.final.model import myConvModel
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p", "-path",  default="sample.jpg")
args = parser.parse_args()
print args
image_path = args.p

CASC_PATH = './haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
SIZE_FACE = 48
EMOTIONS = ['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def brighten(data,b):
     return data * b    

def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
      image,
      scaleFactor = 1.3,
      minNeighbors = 5
  )
  
  # None is we don't found an image
  if not len(faces) > 0:
    print "could not find image"
    return None
  max_area_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
      max_area_face = face
  # Chop image to face
  face = max_area_face
  image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
  # Resize image to network size
  try:
    image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+] Problem during resize")
    return None
  return image

# Load Model
model = myConvModel('models/final/my_model_weights.h5')

def recognise(image_path):
	img=cv2.imread("./samples/"+image_path,0)  
	
	
	img=format_image(img)
	  
	result=None
	
	if img is not None:
		# Predict result with network
		img=img.reshape((1,48,48,1))
		result = model.predict(img)
		print EMOTIONS[np.argmax(result,1)[0]]

recognise(image_path)

while(True):
	image_path = raw_input("Provide image path with respect to ./samples/ directory : ")
	recognise(image_path)


