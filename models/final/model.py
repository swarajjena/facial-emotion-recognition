import os, sys
module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad

#import cv2
import numpy as np


def myConvModel(weights_path=None, shape=(48, 48)):
	model = Sequential()
	
	model.add(Conv2D(32, (5,5), padding='same',input_shape=(48,48,1),activation="relu"))
	model.add(MaxPooling2D((3,3), strides=(2,2)))
	
	model.add(Conv2D(64, (5,5), padding='same',activation="relu"))
	model.add(MaxPooling2D((3,3), strides=(2,2)))

	model.add(Conv2D(128, (4,4), padding='same',activation="relu"))	
	model.add(MaxPooling2D((3,3), strides=(2,2)))
	
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(6, activation='softmax'))
    
	print ("Final Convolutional model created successfully")

	if weights_path:
		model.load_weights(weights_path)
        
	adagrad=Adagrad(lr=0.001)

	model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
#	model.summary()

	return model
