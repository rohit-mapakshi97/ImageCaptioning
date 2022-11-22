#!/usr/bin/python3
import re
import sys 
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical, plot_model
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
# from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

def getPhotoSet(filename):
    photos_list = []
    with open(filename) as file:
        for line in file:
            if len(line) > 1:
                photos_list.append(re.sub('.jpg\n', '', line))
    return set(photos_list)

# load clean descriptions into memory


def loadCleanDescriptions(filename, dataset):
	description_map = {}
	with open(filename) as file: 
		for line in file: 
			if line != '':
				image_id, description = line.replace("\n", "").split("\t")
				if image_id in dataset: 
					if image_id not in description_map:
						description_map[image_id] = []
					if description != '':
						description_map[image_id].append("startseq " + description + " endseq")
	return description_map


# load photo features


def loadPhotoFeatures(filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features

# covert a dictionary of clean descriptions to a list of descriptions


def createTokenizer(descriptions: dict):
    lines = []
    for key in descriptions.keys():
        [lines.append(desc) for desc in descriptions[key]]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(descriptions: dict):
    lines = []
    for key in descriptions.keys():
        [lines.append(desc) for desc in descriptions[key]]
    return max(len(d.split()) for d in lines)

# create sequences of images, input sequences and output words for an image


def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each image identifier
    for key, desc_list in descriptions.items():
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)

# define the captioning model


def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

def getImageFeaturesFile():
	name = ""
	if len(sys.argv) == 1:
		name = "VGG19"
	elif sys.argv[1] == "VGG19":
		name = "VGG19"
	elif sys.argv[1] == "RESNet50":
		name = "RESNet50"
	file_path = "features/image_features_" + name + ".pkl"
	return file_path

if __name__ == "__main__":
    # Load training dataset
	image_features_file = getImageFeaturesFile()
	filename = 'res/Captions/Flickr_8k.trainImages.txt'
	train = getPhotoSet(filename)
	print('Dataset: {}'.format(len(train)))
    # Descriptions
	train_descriptions = loadCleanDescriptions('features/image_descriptions.txt', train)
	print('Descriptions: {}'.format(len(train_descriptions)))
    # photo features
	train_features = loadPhotoFeatures(image_features_file, train)
	print('Photos: train = {}'.format(len(train_features)))
	# prepare tokenizer
	tokenizer = createTokenizer(train_descriptions)
	vocab_size = len(tokenizer.word_index) + 1
	print('Vocabulary Size: {}'.format(vocab_size))
	# determine the maximum sequence length
	max_length = max_length(train_descriptions)
	print('Description Length: {}'.format(max_length))
	# prepare sequences
	X1train, X2train, ytrain = create_sequences(
		tokenizer, max_length, train_descriptions, train_features, vocab_size)

	# Load testing dataset
	filename = 'res/Captions/Flickr_8k.devImages.txt'
	test = getPhotoSet(filename)
	print('Dataset: %d' % len(test))
	# descriptions
	test_descriptions = loadCleanDescriptions('features/image_descriptions.txt', test)
	print('Descriptions: test=%d' % len(test_descriptions))
	# photo features
	test_features = loadPhotoFeatures(image_features_file, test)
	print('Photos: test=%d' % len(test_features))
	# prepare sequences
	X1test, X2test, ytest = create_sequences(
		tokenizer, max_length, test_descriptions, test_features, vocab_size)

	# fit model

	# define the model
	model = define_model(vocab_size, max_length)
	# define checkpoint callback
	filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
	checkpoint = ModelCheckpoint(
		filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	# fit model
	model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[
				checkpoint], validation_data=([X1test, X2test], ytest))
