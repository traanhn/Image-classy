from __future__ import print_function
#simplified interface for building models 
import keras
#because our models are simple
from keras.models import Sequential
#dense means fully connected layers, dropout is a technique to improve convergence, flatten to reshape our matrices for feeding
#into respective layers
from keras.layers import Dense, Dropout, Flatten
#for convolution (images) and pooling is a technique to help choose the most relevant features in an image
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import PIL
import numpy as np
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import model_from_json
from keras import callbacks



train_data_path = './data/train'
validation_data_path = './data/validation'
test_data_path = './data/test'


batch_size = 4
epochs = 15


# Build the model structure 
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
training_set = train_datagen.flow_from_directory(train_data_path,
target_size = (64, 64),
batch_size = batch_size,
class_mode = 'binary')

# Path for test set
test_set = test_datagen.flow_from_directory(test_data_path,
target_size = (64, 64),
batch_size = batch_size,
class_mode = 'binary')

#Path for validation set
validation_set = test_set = test_datagen.flow_from_directory(test_data_path,
target_size = (64, 64),
batch_size = batch_size,
class_mode = 'binary')


classifier.fit_generator(training_set,
steps_per_epoch = 1000,
epochs = epochs,
validation_data = validation_set,
validation_steps = 100)

saved_model = classifier

# how well did it do ?

score = classifier.evaluate(test_set, verbose = 0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])


# Save the model
# Serialize the model to JSON

model_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
	json_file.write(model_json)


# Save model and Serialize weights to HDF5

target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
classifier.save('./models/model.h5')
classifier.save_weights('./models/weights.h5')	






