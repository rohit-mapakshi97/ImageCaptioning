from keras.models import load_model
from pickle import load
from keras.applications.vgg19 import VGG19
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.utils import pad_sequences
from numpy import array, argmax
from IPython.display import Image,display

START_SEQ = "startseq"
END_SEQ = "endseq"

def sampleFeatureExtraction(filename):
    model = VGG19()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

# Generate a description for an image
def generateDescription2(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = START_SEQ
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = wordForId(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == END_SEQ:
            break
    return in_text

# map an integer to a word
def wordForId(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def getSampleImageCaption(filename, model):
    tokenizer = load(open('models/tokenizer_35.pkl', 'rb'))
    photo = sampleFeatureExtraction(filename)
    desc = generateDescription2(model, tokenizer, photo, 35)
    desc = desc.replace('startseq', '')
    desc = desc.replace('endseq', '')
    return desc.title()

if __name__ == "__main__":
    model = load_model('models/-model-ep003-loss3.635-val_loss3.880.h5')
    filename = 'IMG_5125.jpg'
    print(getSampleImageCaption(filename, model))
    display(Image(filename))