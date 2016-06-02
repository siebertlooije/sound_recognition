

#!/usr/bin/env python
#coding: utf-8
""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import os

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)    
    
""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    for root, sub, files in os.walk(audiopath):
        files = sorted(files)
        for f in files:
            print str(f)
            #s_rate, x  =  scipy.io.wavfile.read(os.path.join(root, f))
            print str(root)
            samplerate, samples = wav.read(os.path.join(root, f))
            s = stft(samples, binsize)


            sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
            ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel


            timebins, freqbins = np.shape(ims)



            plt.figure(figsize=(15, 7.5))
            plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
            cur_axes = plt.gca()
	    cur_axes.set_axis_bgcolor('black')
            cur_axes.axes.get_xaxis().set_visible(False)
            cur_axes.axes.get_yaxis().set_visible(False)
           # plt.xlim([0, timebins-1])
           # plt.ylim([0, freqbins])

            #xlocs = np.float32(np.linspace(0, timebins-1, 5))
            #plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
            #ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
            #plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

            #if plotpath:
            #    plt.savefig(plotpath, bbox_inches="tight")
            #else:
            #    plt.show()
            #plt.clf()
            file_name = str(f)
            file_name = file_name.replace(".wav",".jpg")
            root_name = str(root)
            root_name = root_name.replace("ESC-50-wav","figures")
            print str(file_name)
            if not os.path.exists(root_name):
                os.makedirs(root_name)
            plt.savefig(os.path.join(root_name,file_name))


plotstft("/home/siebert/sound_recognition/ESC-50-wav");


