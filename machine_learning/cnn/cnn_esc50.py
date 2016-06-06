from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from PIL import Image
import cPickle as pickle
from keras.utils.np_utils import to_categorical


from random import shuffle

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,108,108)))   #0
    model.add(Convolution2D(64, 3, 3, activation='relu'))       #1
    model.add(ZeroPadding2D((1,1)))                             #2
    model.add(Convolution2D(64, 3, 3, activation='relu'))       #3
    model.add(MaxPooling2D((2,2), strides=(2,2)))               #4

    model.add(ZeroPadding2D((1,1)))                             #5
    model.add(Convolution2D(128, 3, 3, activation='relu'))      #6
    model.add(ZeroPadding2D((1,1)))                             #7
    model.add(Convolution2D(128, 3, 3, activation='relu'))      #8
    model.add(MaxPooling2D((2,2), strides=(2,2)))               #9

    model.add(ZeroPadding2D((1,1)))                             #10
    model.add(Convolution2D(256, 3, 3, activation='relu'))      #11
    model.add(ZeroPadding2D((1,1)))                             #12
    model.add(Convolution2D(256, 3, 3, activation='relu'))      #13
    model.add(ZeroPadding2D((1,1)))                             #14
    model.add(Convolution2D(256, 3, 3, activation='relu'))      #15
    model.add(MaxPooling2D((2,2), strides=(2,2)))               #16

    model.add(ZeroPadding2D((1,1)))                             #17
    model.add(Convolution2D(512, 3, 3, activation='relu'))      #18
    model.add(ZeroPadding2D((1,1)))                             #19
    model.add(Convolution2D(512, 3, 3, activation='relu'))      #20
    model.add(ZeroPadding2D((1,1)))                             #21
    model.add(Convolution2D(512, 3, 3, activation='relu'))      #22
    model.add(MaxPooling2D((1,1), strides=(1,1)))               #23

    model.add(ZeroPadding2D((1,1)))                             #24
    model.add(Convolution2D(512, 3, 3, activation='relu'))      #25
    model.add(ZeroPadding2D((1,1)))                             #26
    model.add(Convolution2D(512, 3, 3, activation='relu'))      #27
    model.add(ZeroPadding2D((1,1)))                             #28
    model.add(Convolution2D(512, 3, 3, activation='relu'))      #29

    #model.add(Flatten())                                        #31
    #model.add(Dense(4096, activation='relu'))                   #32
    #model.add(Dropout(0.5))                                     #33
    #model.add(Dense(4096, activation='relu'))                   #34
    #model.add(Dropout(0.5))                                     #35
    #model.add(Dense(1000, activation='softmax'))                #36

    if weights_path:
        model.load_weights(weights_path)

    return model


def load_esc_coch():
    dataX, dataY = pickle.load(open('../../data/coch_ts.pkl'))

    all_data = zip(dataX, dataY)
    shuffle(all_data)

    dataX, dataY = zip(*all_data)
    dataX = np.asarray(dataX, dtype=np.float64)
    dataY = np.asarray(dataY, dtype='int8')

    print dataX.shape
    dataX = dataX.transpose((0, 3, 1, 2))
    dataX[:, 0, :, :] -= 103.939
    dataX[:, 1, :, :] -= 116.779
    dataX[:, 2, :, :] -= 123.68

    dataX /= np.max(np.abs(dataX))

    dataY = to_categorical(dataY)

    N = dataX.shape[0]

    trainX = dataX[:.8*N]
    trainY = dataY[:.8*N]

    testX = dataX[.8*N:]
    testY = dataY[.8*N:]

    return trainX, trainY, testX, testY


def load_esc():
    dataX, dataY = pickle.load(open('../../data/coch_ts.pkl'))

    dataX = np.asarray(dataX, np.float64)
    dataX[:, 0, :, :] -= 103.939
    dataX[:, 1, :, :] -= 116.779
    dataX[:, 2, :, :] -= 123.68

    dataX /= 255.

    data = zip(dataX, dataY)


    shuffle(data)

    dataX, dataY = zip(*data)
    dataX = np.asarray(dataX)
    dataY = to_categorical(np.asarray(dataY, dtype='int8'))

    trainX, trainY = dataX[:1700], dataY[:1700]
    testX, testY = dataX[1700:], dataY[1700:]

    return trainX, trainY, testX, testY

if __name__ == "__main__":
    print 'loading data...'
    trainX, trainY, testX, testY = load_esc_coch()

    print 'loadig model weights'
    model = VGG_16('vgg16_weights.h5')

    print 'building rest of network'
    model.add(MaxPooling2D((2,2), strides=(2,2)))               #30
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='softmax'))

    print 'compiling model'
    sgd = SGD(lr=0.0001, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    print 'fitting to train data'
    '''
    b_sz = 4
    for epoch in range(20):
        for b_idx in range(trainX.shape[0] / b_sz):
            bX = trainX[b_idx * b_sz:(b_idx+1) * b_sz]
            bY = testY[b_idx * b_sz:(b_idx+1) * b_sz]

            model.train_on_batch(bX, bY)
    '''
    model.fit(trainX, trainY, nb_epoch=100, batch_size=32)
    score = model.evaluate(testX, testY, batch_size=32)

    print 'Done: '
    print score

    #out = model.predict(im)
    #print np.argmax(out)