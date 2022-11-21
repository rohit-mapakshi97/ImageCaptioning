import os 
import sys
from pickle import dump
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.utils import load_img, img_to_array

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

def readDescriptions(filename):
    text = open(filename, 'r')
    file = text.read()
    text.close()

    descriptions = {}
    # Process lines
    for line in file.split('\n'):
        # Split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        imageId = tokens[0]
        imageDesc = tokens[1:]
        # Remove filename from image id
        imageId = imageId.split('.')[0]
        # Convert description tokens back to string
        imageDesc = ' '.join(imageDesc)
        # Create the list
        if imageId not in descriptions:
            descriptions[imageId] = list()
        # Store description
        descriptions[imageId].append(imageDesc)
    return descriptions

def cleanText(captions):
    for image, caption in captions.items():
        for i, imageCaption in enumerate(caption):
            imageCaption.replace("-", " ")
            description = imageCaption.split()
            # Converts to lowercase
            description = [word.lower() for word in description]
            # Remove letters of length 1
            description = [word for word in description if (len(word) > 1)]
            # Remove punctuation
            description = [word for word in description if (word.isalpha())]
            # Convert back to string
            imageCaption = ' '.join(description)
            captions[image][i] = imageCaption
    return captions

def createVocabulary(descriptions):
    vocab = set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab

def saveDescriptions(descriptions, filename):
    lines = list()
    # Build a list of all description strings
    for key, descriptionList in descriptions.items():
        for d in descriptionList:
            lines.append(key + ' ' + d)

    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def split():
    with open("res/trainImages.txt", "w") as a:
        for path, subdirs, files in os.walk('res/images'):
            for filename in files:
                a.write(str(filename) + os.linesep)

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

    #Step 2 Preprocess Text
    dataset = "res/captions.txt"
    filename = dataset

    print("Mapping descriptions dictionary img to 5 captions")
    descriptions = readDescriptions(filename)
    print("Length of descriptions =", len(descriptions))

    print("Cleaning descriptions")
    cleanDescriptions = cleanText(descriptions)

    print("Building vocabulary")
    vocabulary = createVocabulary(cleanDescriptions)

    print("Length of vocabulary = ", len(vocabulary))
    print("Saving each description to file descriptions.txt")
    saveDescriptions(cleanDescriptions, "descriptions.txt")