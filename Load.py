def loadPhotoID(filename):
    file = open(filename, 'r')
    doc = file.read()
    file.close()
    dataset = list()

    for line in doc.split('\n'):
        # get image id
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def loadCleanDescriptions(filename, dataset):
    file = open(filename, 'r')
    doc = file.read()
    file.close()
    descriptions = dict()

    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        imageId, imageDesc = tokens[0], tokens[1:]
        if imageId in dataset:
            # create list
            if imageId not in descriptions:
                descriptions[imageId] = list()
            # Add tags at start & end of description to id start/end of desc
            desc = 'startseq ' + ' '.join(imageDesc) + ' endseq'
            descriptions[imageId].append(desc)
    # return dict of ids to lists of text descriptions
    return descriptions

# load training dataset (~8K)
filename = 'res/trainImages.txt'
train = loadPhotoID(filename)
print('Dataset: %d' % len(train))

# descriptions
trainDescriptions = loadCleanDescriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(trainDescriptions))