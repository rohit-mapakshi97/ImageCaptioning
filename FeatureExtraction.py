from os import listdir
from pickle import dump
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

# extract features from all photos in dir
def extractFeatures(directory):
    # load model
    model = VGG19()
    # restructure model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # summarize
    print(model.summary())
    # extract features from each photo
    features = dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # convert image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # preprocess image for model
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        imageId = name.split('.')[0]
        # store
        features[imageId] = feature
        print('>%s' % name)
    return features

# extract features from all images
directory = 'res/Images'
features = extractFeatures(directory)
print('Extracted Features: %d' % len(features))
dump(features, open('features.pkl', 'wb'))