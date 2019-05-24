
from scipy import signal
from scipy.io import wavfile

import os
import matplotlib.pyplot as plt
'''Below, we plot the spectograms '''

folder = '/home/susovan/Documents/music_detection/muspeak_mirex2015/labeled_seg_callwatch' #folder name containing the .wav files of callwatch
files=[] #stores the names of all full_filename 's
audio_annotation_array=[]
for r, d, f in os.walk(folder): #root, dir, file, still not clear about the efficacy of os.walk
    for file in f:
        if '.wav' in file:
            #below, we make a list of the annotated files
            full_filename=os.path.join(r, file)
            files.append(full_filename)
            if 'm' in file:
                audio_annotation_array.append('m')
            else:
                audio_annotation_array.append('s')
print(audio_annotation_array)


''' plotting the spectograms for music'''
for k in range(15):
    if audio_annotation_array[k] == 'm':
        sample_rate, samples = wavfile.read(files[k])
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        plt.pcolormesh(times, frequencies, spectrogram)
        plt.imshow(spectrogram)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()


