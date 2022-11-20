import os 
import sys 
from pickle import dump
from keras.applications.vgg19 import VGG19
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import Model


# Extract features from all photos in dir
def extractFeatures(model, remove_n_layers, directory):
    # Restructure model
    model = Model(inputs=model.inputs, outputs=model.layers[-remove_n_layers].output)
    # Summarize
    print(model.summary())
    # Extract features from each photo
    features = dict()
    for name in os.listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # Convert image pixels to a numpy array
        image = img_to_array(image)
        # Reshape data for model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # Preprocess image for model
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        imageId = name.split('.')[0]
        # Store
        features[imageId] = feature
        print('>%s' % name)
    return features

# Extract features from all images
if __name__ == "__main__": 
    #Step 1: Extract Image Features
    image_dir = 'res/Images'
    model, remove_n_layers, name  = None, None, None   
    if sys.argv[1] == None:
        model, remove_n_layers, name = VGG19(), 2, "VGG19"
    elif sys.argv[1] == "VGG19":
        model, remove_n_layers, name = VGG19(), 2, "VGG19"
    elif sys.argv[1] == "RESNet50":
        # model, remove_n_layers = **, some_no, "RESNet50"
        pass 
    
    features = extractFeatures(VGG19(), 2, image_dir)
    print('Extracted Features: %d' % len(features))
    file_path = "features/image_features" + name + 'pkl'
    dump(features, open(file_path, 'wb'))

    #Step 2: Extract Text Features 
    #Step 2.1 Preprocess Text 
    #Step 2.2 Extract Text Features   
