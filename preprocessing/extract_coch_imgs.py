import h5py
import plot


#you can use your own config, make sure you create a 'config.py' file with a 'models' array inside
import pycpsp.sourceindicators.config as config
import pycpsp.sourceindicators.util as util
import pycpsp.files as files
import pycpsp.plot as plot
import os, matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize

import cPickle as pickle


root = '../data/ESC-50-wav/resampled'
dirs = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
dirs.sort()

#f, axarr = plt.subplots(4, sharex=True)

label = 0
images = []
labels = []
for subdir in dirs:
    d = os.path.join(root, subdir)
    hdf5dir = os.path.join(d, 'ptne/hdf5/2016-06-02')

    h5files = [f for f in os.listdir(hdf5dir) if '.wav.3' in f]

    if 'Pouring' in hdf5dir or 'Church bells' in hdf5dir or 'Glass' in hdf5dir:
        print 'skipping ', hdf5dir
        continue

    print 'Working on ', hdf5dir
    for h5file in h5files:
        signals = files.signalsFromHDF5(os.path.join(hdf5dir, h5file))
        energy = signals['energy']
        pulse = signals['tfmax']
        tone = signals['tsmax']

        max = np.max([tone, energy, pulse])
        min = np.min([tone, energy, pulse])

        full_im = (np.stack([tone, pulse, energy], axis=0).transpose((1, 2, 0)) - min) / (max - min)
        full_im = imresize(full_im, (109, 115))
        images.append(full_im)
        labels.append(label)

        #plt.imshow(255 - full_im)
        #plt.pause(0.05)
    label += 1


f = open('chochs.pkl', 'wb')
pickle.dump([images, labels], f)




'''
path = '../data/ESC-50-wav/resampled/101 - Dog/ptne/hdf5/2016-06-02/3-157695-A.wav.1.hdf5'
filepointer = h5py.File(path, 'r')

keys = filepointer.keys()
attrs = filepointer.attrsprint dir
print attrs['inputrate']
print """
Keys: {}
Metadata: {}
Signals:
{}""".format(
', '.join(keys),
', '.join([key for key in attrs]),
[filepointer[key] for key in keys]
)


signals = files.signalsFromHDF5(path)

energy = signals['energy']
'''

#plot.plot2D('Energy', energy)