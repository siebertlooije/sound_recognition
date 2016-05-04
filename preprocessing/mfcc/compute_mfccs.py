from scipy.io import wavfile
import os
import mfcc

import matplotlib.pyplot as plt

path = '../../data/ESC-50-wav'

fig = plt.figure(frameon=False)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')


def save_mfcc_im(image, path):
    plt.imshow(image)
    fig.canvas.print_png(path)
    #frame = plt.gca()
    #frame.axes.get_xaxis().set_visible(False)
    #frame.axes.get_yaxis().set_visible(False)
    #plt.savefig(path, bbox_inches='tight', pad_inches=0.0)
    #plt.show()


def compute_mfccs(demo=True):
    for dir in os.listdir( path ):
        dir_path = os.path.join(path, dir)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            try:
                rate, data = wavfile.read(file_path)
            except ValueError:
                print 'Not a WAV file: ', file_path
                continue
            #print len(data.shape)
            if len(data.shape) is 1:
                mfcc_features, mfcc_im = mfcc.extract(data, show=demo)
                if demo:
                    return
                print 'saving:', file_path.replace('.wav', '.png')
                save_mfcc_im(mfcc_im, file_path.replace('.wav', '.png'))
            else:
                mfcc_features_l, mfcc_im_l = mfcc.extract(data[:, 0], show=demo)
                mfcc_features_r, mfcc_im_r = mfcc.extract(data[:, 1], show=demo)

                save_mfcc_im(mfcc_im_l, file_path.replace('.wav', '_l.png'))
                save_mfcc_im(mfcc_im_r, file_path.replace('.wav', '_r.png'))



if __name__=="__main__":
    compute_mfccs(demo=False)