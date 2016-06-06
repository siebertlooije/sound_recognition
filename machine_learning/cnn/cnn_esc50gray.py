from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta
import cv2, numpy as np
from PIL import Image
import cPickle as pickle
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

from random import shuffle


from random import shuffle

idx_to_label = ['Dog', 'Rooster', 'Pig', 'Cow', 'Frog', 'Cat',
                'Hen', 'Insects', 'Sheep', 'Crow', 'Rain', 'Sea waves',
                'Crackling fire', 'Crickets', 'Chirping birds',
                'Water drops', 'Wind', 'Pouring water', 'Toilet flush',
                'Thunderstorm', 'Crying baby', 'Sneezing', 'Clapping',
                'Breathing', 'Coughing', 'Footsteps', 'Laughing',
                'Brushing teeth', 'Snoring', 'Drinking - sipping'
                'Door knock', 'Mouse click', 'Keyboard typing',
                'Door - wood creaks', 'Can opening', 'Washing machine',
                'Vacuum cleaner', 'Clock alarm', 'Clock tick'
                'Glass breaking', 'Helicopter', 'Chainsaw', 'Siren', 'Car Horn',
                'Engine', 'Train', 'Curch bells', 'Airplane', 'Fireworks',
                'Hand saw']

def load_esc():
    trainX, trainY = pickle.load(open('../../data/esc50_train.pkl'))
    testX, testY = pickle.load(open('../../data/esc50_test.pkl'))

    trainY = to_categorical(np.asarray(trainY, dtype='int8'))

    test_labels = np.asarray(testY, dtype='int8')
    testY = to_categorical(np.asarray(testY, dtype='int8'))
    '''
    print max(dataY)

    dataX = dataX.reshape((dataX.shape[0], 1, dataX.shape[1], dataX.shape[2]))

    data = zip(dataX, dataY)
    shuffle(data)

    dataX, dataY = zip(*data)
    dataX = np.asarray(dataX)
    dataY = to_categorical(np.asarray(dataY, dtype='int8'))

    trainX, trainY = dataX[:1700], dataY[:1700]
    testX, testY = dataX[1700:], dataY[1700:]
    '''

    return trainX.reshape((trainX.shape[0], 1, 108, 108)) / 255., trainY, testX.reshape(testX.shape[0], 1, 108, 108) / 255., testY, test_labels

def load_esc_coch():
    dataX, dataY = pickle.load(open('../../data/coch_ts.pkl'))

    all_data = zip(dataX, dataY)
    shuffle(all_data)

    dataX, dataY = zip(*all_data)
    dataX = np.asarray(dataX, dtype=np.float64)
    dataY = np.asarray(dataY, dtype='int8')

    print dataX.shape
    dataX = dataX.transpose((0, 3, 1, 2))

    data_mean = np.mean(dataX, axis=(0, 2, 3))

    #dataX[:, 0,:,:] -= data_mean[0]
    #dataX[:, 1,:,:] -= data_mean[1]
    #dataX[:, 2,:,:] -= data_mean[2]

    dataX = dataX[:, -1:, :, :]

    dataX /= np.max(np.abs(dataX))

    test_labels = dataY

    dataY = to_categorical(dataY)

    N = dataX.shape[0]

    trainX = dataX[:.8*N]
    trainY = dataY[:.8*N]

    testX = dataX[.8*N:]
    testY = dataY[.8*N:]

    return trainX, trainY, testX, testY, test_labels

if __name__ == "__main__":
    print 'loading data...'
    cochs = False
    load_w = True

    if cochs:
        trainX, trainY, testX, testY, test_labels = load_esc_coch()
    else:
        trainX, trainY, testX, testY, test_labels = load_esc()

    model = Sequential()

    print 'building rest of network'
    model.add(Convolution2D(32, 7, 7, activation='relu', input_shape=(trainX.shape[1], 108, 108)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Convolution2D(64, 5, 5, activation='relu'))
    model.add(Convolution2D(64, 5, 5, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='softmax'))

    print 'compiling model'

    if load_w:
        model.load_weights('cnn_homemade_coch.h5' if cochs else 'cnn_homemade.h5')

    adadelta = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(optimizer=adadelta, loss='categorical_crossentropy', metrics=['accuracy'])
    if load_w:
        errors = 0
        total = 0.
        confusion_matrix = np.zeros((50, 50))
        for iter in range(len(testX)):
            pred = model.predict_classes(testX[iter:iter+1], batch_size=1)
            confusion_matrix[test_labels[iter], pred[0]] += 1

            errors += (test_labels[iter] != pred[0])
            total += 1
        print 'Test error: ', errors / total

        pickle.dump(confusion_matrix, open('confusion_matrix.pkl', 'wb'))
        plt.imshow(confusion_matrix)
        plt.show()


    if cochs:
        datagen = ImageDataGenerator(
            width_shift_range=.2,
            height_shift_range=.1
        )
        datagen.fit(trainX)
        print 'fitting to train data'
        for _ in range(50):
            model.fit_generator(datagen.flow(trainX, trainY, batch_size=32), samples_per_epoch=len(trainX), nb_epoch=100)
            model.save_weights('cnn_homemade_coch.h5', overwrite=True)
            score = model.evaluate(testX, testY, batch_size=32)
            print 'Test accuracy: ', score[1]
    else:
        print 'fitting to train data'
        for _ in range(10):
            model.fit(trainX, trainY, batch_size=32, nb_epoch=5)
            model.save_weights('cnn_homemade.h5', overwrite=True)
            score = model.evaluate(testX, testY, batch_size=32)
            print 'Test accuracy: ', score[1]

    print 'Done: '
    print score

    #out = model.predict(im)
    #print np.argmax(out)