# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:34:20 2018

@author: DGC user
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 18:21:14 2018

@author: Jani Rantanen
"""
# https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import PIL
import numpy as np
from keras.layers import Activation, Dropout, Flatten, Dense

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64,64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
classifier.add(Dense(64))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(1))
classifier.add(Activation('sigmoid'))

classifier.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Path for train set
training_set = train_datagen.flow_from_directory('E:/Altia/test1/train/',#'C:/Users/DGC User/OneDrive - Louhia Analytics Oy/Hack/test1/train/',
target_size = (64, 64),
batch_size = 4,
class_mode = 'binary')

# Path for test set
test_set = test_datagen.flow_from_directory('E:/Altia/test1/test/',#'C:/Users/DGC User/OneDrive - Louhia Analytics Oy/Hack/test1/test',
target_size = (64, 64),
batch_size = 4,
class_mode = 'binary')

classifier.fit_generator(training_set,
steps_per_epoch = 1000,
epochs = 15,
validation_data = test_set,
validation_steps = 100)
saved_model = classifier

#################### Test with test image without service
from keras.preprocessing import image
test_image = image.load_img('C:/Users/DGC User/OneDrive - Louhia Analytics Oy/Hack/test1/valid/test.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
 prediction = 'notok'
else:
 prediction = 'ok'

prediction
 result = classifier.predict_proba(test_image)
 ########################
 
 
 ##########################################################
 ####Service
 
from bottle import route, run, template

@route('/Altia/<name>')
def index(name):
    image_test='E:/Altia/test1/valid/' + name + '.jpg'
    test_image = image.load_img(image_test, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image/255
    result = classifier.predict(test_image)
    if result[0][0] == 1:
     prediction = 'notOK'
     prediction_prob = np.array2string(classifier.predict_proba(test_image))
    else:
     prediction = 'OK'
     prediction_prob = np.array2string(classifier.predict_proba(test_image))
    return 'This bottle is ' + prediction + '. The probability to be notOK: ' + prediction_prob

run(host='localhost', port=8080)




 