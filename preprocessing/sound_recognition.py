import warnings
warnings.filterwarnings('ignore')

# numerical processing and scientific libraries
import numpy as np
import scipy
import wave

# signal processing
from scipy.io                     import wavfile
from scipy                        import stats, signal
from scipy.fftpack                import fft

from scipy.signal                 import lfilter, hamming
from scipy.fftpack.realtransforms import dct
from scikits.talkbox              import segment_axis
from scikits.talkbox.features     import mfcc

# general purpose
import collections

# plotting
import matplotlib.pyplot as plt
from numpy.lib                    import stride_tricks

from base64                       import b64encode

# Classification and evaluation
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd

#samplerate, wavedata = wavfile.read("/home/siebert/sound_recognition/dataset/test.wav")
##number_of_samples = wavedata.shape[0]
#song_length = int(number_of_samples/samplerate)
#show_stereo_waveform(wavedata);
wave.open("/home/siebert/sound_recognition/dataset/test.wav","r");
(nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams ()
print framerate;