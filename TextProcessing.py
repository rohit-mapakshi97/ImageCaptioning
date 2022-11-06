# Prepare Text Data

def readDescriptions(filename):
    text = open(filename, 'r')
    file = text.read()
    text.close()

    descriptions = {}
    # process lines
    for line in file.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        imageId = tokens[0]
        imageDesc = tokens[1:]
        # remove filename from image id
        imageId = imageId.split('.')[0]
        # convert description tokens back to string
        imageDesc = ' '.join(imageDesc)
        # create the list
        if imageId not in descriptions:
            descriptions[imageId] = list()
        # store description
        descriptions[imageId].append(imageDesc)
    return descriptions

def cleanText(captions):
    for image, caption in captions.items():
        for i, imageCaption in enumerate(caption):
            imageCaption.replace("-", " ")
            description = imageCaption.split()
            # converts to lowercase
            description = [word.lower() for word in description]
            # remove letters of length 1
            description = [word for word in description if (len(word) > 1)]
            # remove punctuation
            description = [word for word in description if (word.isalpha())]
            # convert back to string
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
    # build a list of all description strings
    for key, descriptionList in descriptions.items():
        for d in descriptionList:
            lines.append(key + '\t' + d)

    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

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