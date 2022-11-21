from numpy import array
from pickle import load
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.utils import pad_sequences, to_categorical, plot_model

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

if __name__ == "__main__":
    # Step 1 Load Data
    # Load training dataset (~6K)
    filename = 'res/trainImages.txt'
    train = loadPhotoID(filename)
    print('Dataset: %d' % len(train))

    trainDescriptions = loadCleanDescriptions('descriptions.txt', train)
    print('Descriptions: train=%d' % len(trainDescriptions))

    trainFeatures = loadPhotoFeatures('features.pkl', train)
    print('Photos: train=%d' % len(trainFeatures))

    # Step 2 Prepare Tokenizer
    tokenizer = createTokenizer(trainDescriptions)
    vocabSize = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocabSize)

    # Step 3 Encode Text
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
    model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint],
              validation_data=([X1test, X2test], ytest))
