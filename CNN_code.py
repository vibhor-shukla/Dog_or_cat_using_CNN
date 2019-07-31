
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()
#adding Convlotion layer with 32 channels of size=(3,3) and size of input layer should be (width X height X 3{for colored picture})
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),strides=1,activation='relu',use_bias=False,bias_initializer='zeros',input_shape=(64,64,3)))
#adding maxpooling layers

classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#flattening layers
classifier.add(Flatten())
# adding fully connected layers and units means output dim
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

#now compiling classifier
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#now preprocessing images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
#import keras

classifier.fit_generator(train_set,
                         steps_per_epoch = 140,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 70)

#making new prediction
import numpy as np
train_set.class_indices
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
if result[0][0] is 1:
    pred1='dog'
else:
    pred1='cat'

test_image1 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64,64))
test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1,axis=0)
result = classifier.predict(test_image1)
if result[0][0] is 1:
    pred2='dog'
else:
    pred2='cat'  
