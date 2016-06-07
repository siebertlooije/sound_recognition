import matplotlib.pyplot as plt
import cPickle as pkl

import numpy as np

M = pkl.load(open('confusion_matrix.pkl', 'r'))


index = np.arange(50)
bar_width = 0.35


idx_to_label = \
    ['Dog',
     'Rooster',
     'Pig',
     'Cow',
     'Frog',
     'Cat',
     'Hen',
     'Insects',
     'Sheep',
     'Crow',
     'Rain',
     'Sea waves',
     'Crackling fire',
     'Crickets',
     'Chirping birds',
     'Water drops',
     'Wind',
     'Toilet flush',
     'Thunderstorm',
     'Crying baby',
     'Sneezing',
     'Clapping',
     'Breathing',
     'Coughing',
     'Footsteps',
     'Laughing',
     'Brushing teeth',
     'Snoring',
     'Drinking - sipping',
     'Door knock',
     'Mouse click',
     'Keyboard typing',
     'Door - wood creaks',
     'Can opening',
     'Washing machine',
     'Vacuum cleaner',
     'Clock alarm',
     'Clock tick',
     'Glass breaking',
     'Helicopter',
     'Chainsaw',
     'Siren',
     'Car Horn',
     'Engine',
     'Airplane',
     'Fireworks',
     'Hand saw']

print len(idx_to_label)


M = M.transpose()


d = np.diag(M)
r = (np.sum(d) - d[17] - d[45]) / (np.sum(M) - np.sum(M[17]) - np.sum(45))

print 'rate without pouring water: ', r


for idx, row in enumerate(M):

    plt.clf()
    plt.bar(index, row, bar_width)
    plt.xlabel('Target label')
    plt.title(idx_to_label[idx])
    plt.xticks(index + bar_width, idx_to_label, rotation='vertical')

    plt.savefig('im/t' + str(idx) + idx_to_label[idx] + '.png')

M = M.transpose()

for idx, row in enumerate(M):

    plt.clf()
    plt.bar(index, row, bar_width)
    plt.xlabel('Predicted label')
    plt.title(idx_to_label[idx])
    plt.xticks(index + bar_width, idx_to_label, rotation='vertical')

    plt.savefig('im/' + str(idx) + idx_to_label[idx] + '.png')




