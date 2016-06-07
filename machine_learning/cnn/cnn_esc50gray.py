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

    return trainX.reshape((trainX.shape[0], 1, 109, 115)) / 255., trainY, testX.reshape(testX.shape[0], 1, 109, 115) / 255., testY, test_labels

def load_esc_coch():
    dataX, dataY = pickle.load(open('../../preprocessing/chochs.pkl'))

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
    cochs = True
    load_w = False

    if cochs:
        trainX, trainY, testX, testY, test_labels = load_esc_coch()
    else:
        trainX, trainY, testX, testY, test_labels = load_esc()

    model = Sequential()

    print 'building rest of network'
    model.add(Convolution2D(32, 7, 7, activation='relu', input_shape=(trainX.shape[1], 109, 115))) # 102 x 102
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))         # 51 x 51
    model.add(Convolution2D(64, 5, 5, activation='relu'))   # 47 x 47
    model.add(Convolution2D(64, 5, 5, activation='relu'))   # 43 x 43
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))         # 21 x 21
    model.add(Convolution2D(128, 3, 3, activation='relu'))  # 19 x 19
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))         # 19 x 1

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(trainY.shape[1], activation='softmax'))

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

    best_score = 0
    if cochs:
        datagen = ImageDataGenerator(
            width_shift_range=.2,
            height_shift_range=.2,
            fill_mode='constant'
        )
        datagen.fit(trainX)
        print 'fitting to train data'
        for _ in range(50):
            model.fit_generator(datagen.flow(trainX, trainY, batch_size=32, shuffle=True),
                                samples_per_epoch=len(trainX), nb_epoch=100,
                                verbose=0)
            _, score = model.evaluate(testX, testY, batch_size=32, verbose=0)
            if score > best_score:
                print 'New best score!'
                best_score = score
                model.save_weights('cnn_homemade_coch.h5', overwrite=True)
            print 'Test accuracy: ', score
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