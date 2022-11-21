import os 
import sys
import pandas as pd
from numpy import array
from pickle import dump, load
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.utils import load_img, img_to_array, pad_sequences, to_categorical, plot_model



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
            lines.append(key + '\t' + d)

    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def loadPhotoID(filename):
    file = open(filename, 'r')
    doc = file.read()
    file.close()
    dataset = list()

    for line in doc.split('\n'):
        # Get image id
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def loadCleanDescriptions(filename, dataset):
    file = open(filename, 'r')
    doc = file.read()
    file.close()
    descriptions = dict()

    for line in doc.split('\n'):
        # Split line by white space
        tokens = line.split()
        imageId, imageDesc = tokens[0], tokens[1:]
        if imageId in dataset:
            # Create list
            if imageId not in descriptions:
                descriptions[imageId] = list()
            # Add tags at start & end of description to id start/end of desc
            desc = 'startseq ' + ' '.join(imageDesc) + ' endseq'
            descriptions[imageId].append(desc)
    # Return dict of ids to lists of text descriptions
    return descriptions

def loadPhotoFeatures(filename, dataset):
    allFeatures = load(open(filename, 'rb'))
    features = {k: allFeatures[k] for k in dataset}
    return features

def toLines(descriptions):
    all = list()
    for key in descriptions.keys():
        [all.append(d) for d in descriptions[key]]
    return all

def createTokenizer(descriptions):
    lines = toLines(descriptions)
    t = Tokenizer()
    t.fit_on_texts(lines)
    return t

def createSequences(tokenizer, maxLength, descriptions, photos, vocabSize):
    X1 = list()
    X2 = list()
    y = list()

    for key, descList in descriptions.items():
        for desc in descList:
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                inSeq, outSeq = seq[:i], seq[i]
                # pad input sequence
                inSeq = pad_sequences([inSeq], maxlen=maxLength)[0]
                # encode output sequence
                outSeq = to_categorical([outSeq], num_classes=vocabSize)[0]
                # store
                X1.append(photos[key][0])
                X2.append(inSeq)
                y.append(outSeq)
    return array(X1), array(X2), array(y)

def defineModel(vocabSize, maxLength):
    # feature extractor model
    input1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(input1)
    fe2 = Dense(256, activation='relu')(fe1)
    # Sequence model
    input2 = Input(shape=(maxLength,))
    se1 = Embedding(vocabSize, 256, mask_zero=True)(input2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(Dropout(0.5)(se2))
    # Decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocabSize, activation='softmax')(decoder2)
    # [image, seq] [word]
    model = Model(inputs=[input1, input2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

def maxLength(descriptions):
    lines = toLines(descriptions)
    return max(len(d.split()) for d in lines)

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

    #Step 3 Load Data
    # Load training dataset (~6K)
    filename = 'res/trainImages.txt'
    train = loadPhotoID(filename)
    print('Dataset: %d' % len(train))

    trainDescriptions = loadCleanDescriptions('descriptions.txt', train)
    print('Descriptions: train=%d' % len(trainDescriptions))

    trainFeatures = loadPhotoFeatures('features.pkl', train)
    print('Photos: train=%d' % len(trainFeatures))

    #Step 2.2 Prepare Tokenizer
    tokenizer = createTokenizer(trainDescriptions)
    vocabSize = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocabSize)

    #Step 3 Encode Text
    # prepare sequences
    # determine the maximum sequence length
    maxlength = maxLength(trainDescriptions)
    print('Description Length: %d' % maxlength)
    # prepare sequences
    X1train, X2train, ytrain = createSequences(tokenizer, maxlength, trainDescriptions, trainFeatures, vocabSize)

    # load test set
    filename = 'res/testImages.txt'
    test = loadPhotoID(filename)
    print('Dataset: %d' % len(test))
    # descriptions
    testDescriptions = loadCleanDescriptions('descriptions.txt', test)
    print('Descriptions: test=%d' % len(testDescriptions))
    # photo features
    testFeatures = loadPhotoFeatures('features.pkl', test)
    print('Photos: test=%d' % len(testFeatures))
    # prepare sequences
    X1test, X2test, ytest = createSequences(tokenizer, maxLength, testDescriptions, testFeatures, vocabSize)

    # define the model
    model = defineModel(vocabSize, maxLength)
    # define checkpoint callback
    filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # fit model
    model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint],
              validation_data=([X1test, X2test], ytest))
