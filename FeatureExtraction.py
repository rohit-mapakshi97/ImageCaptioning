import os
import sys
import pickle
from keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet import preprocess_input
from keras.models import Model
import pandas as pd
import re

def extractFeatures(model, remove_n_layers: int, directory: str) -> dict:
    # Restructure model
    model = Model(inputs=model.inputs,
                  outputs=model.layers[-remove_n_layers].output)
    # Summarize
    print(model.summary())
    # Extract features from each photo
    features = {}
    for name in os.listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # Convert image pixels to a numpy array
        image = img_to_array(image)
        # Reshape data for model
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        # Preprocess image for model
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        imageId = name.split('.')[0]
        # Store
        features[imageId] = feature
        print('>%s' % name)
    return features

def preprocessText(line: str):
    # 1 Lowercase & Remove special charecters
    REGEX_SPECIAL_CHARS = '[^A-Za-z0-9]+'
    line = re.sub(REGEX_SPECIAL_CHARS, ' ', line.lower()).strip()
    # Remove words with numbers
    tokens = line.split()
    tokens = [w for w in tokens if w.isalpha()]
    # Remove single-word letters
    tokens = [word for word in tokens if len(word) > 1]
    line = " ".join(tokens)
    #  Change multiple spaces to one space
    REGEX_SPACE = '\s+'
    line = re.sub(REGEX_SPACE, ' ', line)
    return line

def removeExtension(line: str):
    REGEX = '.jpg#[0-9]'
    line = re.sub(REGEX, '', line)
    return line

def getModelConfiguration(): 
    model, remove_n_layers, name = None, None, None
    if len(sys.argv) == 1:
        model, remove_n_layers, name = VGG19(), 2, "VGG19"
    elif sys.argv[1] == "VGG19":
        model, remove_n_layers, name = VGG19(), 2, "VGG19"
    elif sys.argv[1] == "ResNet50":
        model, remove_n_layers, name = ResNet50(), 2, "ResNet50"
    return model, remove_n_layers, name

if __name__ == "__main__":
    # Step 1: Extract Image Features
    
    model, remove_n_layers, name = getModelConfiguration()
    image_dir = "res/Images"
    file_path = "features/image_features_" + name + ".pkl"
    if not (os.path.isfile(file_path)):
        features = extractFeatures(model, 2, image_dir)
        print("Extracted Features: {}".format(len(features)))
        pickle.dump(features, open(file_path, "wb"))

    # Step 2: Preprocess Captions
    captions_file = "res/Captions/Flickr8k.token.txt"
    descriptions_file = "features/image_descriptions.txt"
    if not (os.path.isfile(descriptions_file)):
        df = pd.read_csv(captions_file, sep="\t", header=None)
        df[0] = df.apply(lambda x: removeExtension(x[0]), axis=1)
        df[1] = df.apply(lambda x: preprocessText(x[1]), axis=1)
        df.to_csv(descriptions_file, index=False, sep="\t", header=None)
