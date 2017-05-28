from __future__ import print_function

import numpy as np

from keras.callbacks import LambdaCallback, EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from models.final.model import myConvModel



def main():
    model = myConvModel('models/final/my_model_weights.h5')

    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')

    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    print(X_train.shape)
    print(y_train.shape) 
   
    print("Training started")

    callbacks = []
    earlystop_callback = EarlyStopping(monitor='val_loss', patience=30, verbose=0)
    checkpoint_callback = ModelCheckpoint(filepath='models/final/checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5')

    callbacks.append(earlystop_callback)
    callbacks.append(checkpoint_callback)

    batch_size = 512

    train_datagen = ImageDataGenerator(
		rescale=1./255,
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		zoom_range=0.2,				
		horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    # compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied)
    train_datagen.fit(X_train)
    test_datagen.fit(X_test)

    epochs=400
    batch_size=128

	# fits the model on batches with real-time data augmentation:
    model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size), \
						len(y_train) / batch_size, \
						epochs, \
    					callbacks=callbacks,\
    					validation_data=test_datagen.flow(X_test, y_test, batch_size=batch_size),\
						validation_steps=len(y_test) / batch_size\
						)


    model.save_weights('models/final/my_model_weights.h5')
    scores = model.evaluate((X_test*(1.0/255.0)), y_test, verbose=0)
    print ("Test loss : %.3f" % scores[0])
    print ("Test accuracy : %.3f" % scores[1])
    print ("Testing finished")

if __name__ == "__main__":
    main()
