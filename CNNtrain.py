#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


def getEncoderModel(standard_size: tuple, channels: int): 
    # Initialing the CNN
    encoder = Sequential()

    kernel_size = (3,3) # Size of kernel matrix used for extracting high level features 
    filters_1 = 32 # number of filters the convolution layer learns and number of output filters 
    filters_n = 64 # number of filters the convolution layer learns and number of output filters 
    pool_size = (2,2) # Pooling for dimensionality reduction 

    # Adding 1st convolution Layer 
    encoder.add(Convolution2D(filters_1, kernel_size, input_shape = (standard_size[0], standard_size[1], channels), activation = "relu"))
    encoder.add(MaxPooling2D(pool_size))

    # Adding 2nd convolution layer
    encoder.add(Convolution2D(filters_n, kernel_size, activation = "relu"))
    encoder.add(MaxPooling2D(pool_size))

    #Adding 3rd convolution Layer
    encoder.add(Convolution2D(filters_n, kernel_size, activation = "relu"))
    encoder.add(MaxPooling2D(pool_size))

    #Step 3 - Flattening will convert the 3D output to 1D output 
    encoder.add(Flatten())

    #Todo add one dense layer based on the image vector size that you need 
    encoder.add(Dense(64, activation='relu'))

    print(encoder.summary())
    #Compiling The CNN Encoder 
    # encoder.compile(
    #             optimizer = optimizers.SGD(lr = 0.01),
    #             loss = 'categorical_crossentropy',
    #             metrics = ['accuracy'])

#Part 2 Fittting the CNN to the image

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# training_set = train_datagen.flow_from_directory(
#         'data2/training_set',
#         target_size=(64, 64),
#         batch_size=32,
#         class_mode='categorical')

# test_set = test_datagen.flow_from_directory(
#         'data2/test_set',
#         target_size=(64, 64),
#         batch_size=32,
#         class_mode='categorical')

# model = classifier.fit_generator(
#         training_set,
#         steps_per_epoch=800,
#         epochs=25,
#         validation_data = test_set,
#         validation_steps = 6500
#       )

'''#Saving the model
import h5py
classifier.save('Trained_model.h5')'''

# print(model.history.keys())
# import matplotlib.pyplot as plt
# # summarize history for accuracy
# plt.plot(model.history['acc'])
# plt.plot(model.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss

# plt.plot(model.history['loss'])
# plt.plot(model.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

if __name__ == "__main__": 
    # Part 1 - Building the CNN
    standard_size = (224, 224) #image size 
    channels = 3 #RGB channels 
    encoder = getEncoderModel(standard_size, channels); 

