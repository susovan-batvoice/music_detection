'''Below we segment a wave file into segments of specified time durations, which come from
the time durations of non-overlapping time intervals of music and speech. Then we make
the MFCC coeffs. from these intervals '''



from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pydub import AudioSegment
import matplotlib.pyplot as plt
import scipy
import numpy as np
from convert_time_overlapping_df_into_time_non_overapping_df import make_non_time_overlapping_df_from_time_overlapping_df
from slice_sound import make_sound_chunks_from_df_start_endtime_annotation_chunks, make_time_chunks_from_df_of_start_end_times
import os
from make_df_et_matrix import df_et_matrix_one_file
from slice_sound import make_sound_chunks_from_df_start_endtime_annotation_chunks


np.set_printoptions(precision=1000)


def resample_matrix(F,num):
    G=np.empty((F.shape[0], num))
    for i in range(F.shape[0]):
      G[i]=scipy.signal.resample(F[i], num, t=None, axis=0, window=None)
    return G

def build_MFCC_for_one_sound_slice(folder,sound_slice):
    '''
    builds the MFCC coeffs, given the sound_slice and folder containing the sound_slice
    :param file: str
    :param sound_slice:
    :return: array
    folder is the folder name where sound_slice is
    sound_slice must be in .wav format, not .mp3 format, in order to apply pyAusioAnalysis
    '''

    sound_slice_fullname = os.path.join(folder,sound_slice)
    [Fs, x] = audioBasicIO.readAudioFile(sound_slice_fullname)
    F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
    G = resample_matrix(F, num=15)  # see the definitionog resample_matrix above
    #feat_list=[]
    #feat_list.append(G)
    '''Below, we extract the MFCC's from the extracted and then resampled features above 
    .For each i, MFCC_list[i] gives for the i-th audio, all 13 feature vectors, each of them 15-dimensional 
    resampled values of the original feature vectors'''
    MFCC_list = []
    #for i in range(len(feat_list)):
    #MFCC = feat_list[i][8:21, :]  # 9th to 21-st features are the MFCC coeffs
    MFCC = G[8:21, :]  # 9th to 21-st features are the MFCC coeffs
    MFCC_flat = np.ndarray.flatten(MFCC)  # flatening the array, but are we destroying time series structure?
    MFCC_flattned_as_list=list(MFCC_flat) #MFCC_array was np array, so we convert it into list, to avoid dim like [1, foo, bar]
    #MFCC_list.append(MFCC_flat_as_list)
    #MFCC_array = np.asarray(MFCC_list)
    return MFCC_flattned_as_list



def build_MFCC_for_audio_seg_list(folder,chunk_list, audio_annotation_list):
    '''

    :param folder: str, folder name
    :param chunk_list: list of str of all audio samples we wish to build MFCC of, it's the output of the function
    make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[0]
    :param audio_annotation_list: list of corresponding audio annotations (music, speech) for the above sound chunks
    :return: N*d dim array of all corresponding d-dim MFCC coeffs, where N= no of chunks in chunk_list, np array
    of the audio_annotation_list
    '''

    MFCC_list_for_all_audio_segs=[]
    #for i in range(1):
    for i in range(len(chunk_list)):
        sound_slice=chunk_list[i] #i-th sound slice in the list, it's a str
        MFCC_sound_slice=build_MFCC_for_one_sound_slice(folder,sound_slice)
        MFCC_list_for_all_audio_segs.append(MFCC_sound_slice)
        #MFCC_array=np.asarray(MFCC_list_for_all_audio_segs)
        MFCC_array = np.array(MFCC_list_for_all_audio_segs)
        tmp_shape=MFCC_array.shape
        audio_annotation_array = np.array(audio_annotation_list)
        tmp_shape2=audio_annotation_array.shape
        foo='give a breakpoint here'
    return MFCC_array, audio_annotation_array



'''
#TEST the code build_MFCC_for_one_sound_slice above:
folder="/home/susovan/Documents/music_detection/muspeak_mirex2015/music_chunks"
sound_slice="sound-1.wav"
MFCC_sound_slice=build_MFCC_for_one_sound_slice(folder,sound_slice)
print("\n The MFCC for " + str(sound_slice) + " is: \n " + str(MFCC_sound_slice) )
print("\n The flattened MFCC_list has length:     \n" + str(len(MFCC_sound_slice))   )
'''



'''
#TEST the code build_MFCC_for_audio_seg_list above:

#folder="/home/susovan/Documents/music_detection/muspeak_mirex2015/music_chunks"
#sound_slice="sound-174.wav"
folder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'
file1='/ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.csv'
file_mp3=  '/ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.mp3'
df=df_et_matrix_one_file(folder,file1)[0]
df_nonoverlapping=make_non_time_overlapping_df_from_time_overlapping_df(df)
#print("df_nonoverlapping is \n" + str(df_nonoverlapping))
df_start_endtime_annotation_chunks= make_time_chunks_from_df_of_start_end_times(df_nonoverlapping, chunk_dur=10)
audio= os.path.join(folder, file_mp3)
audio= folder + file_mp3
#audio = AudioSegment.from_file(audio, format="mp3", chennels=1)
#audio = AudioSegment.from_file(audio, format="wav", chennels=1)
audio_seg_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[0]
audio_annotation_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[1]
chunk_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[2]
#print(  "\n df_start_endtime_annotation_chunks is \n" + str(df_start_endtime_annotation_chunks) )
#print( "\n audio_seg_list \n"    + str(audio_seg_list) )
#print( "\n audio_annotation_list \n" + str(audio_annotation_list) )
#print("\n chunk_list is \n" + str(chunk_list) )
tmp=build_MFCC_for_audio_seg_list(folder,chunk_list, audio_annotation_list)
MFCC_array = tmp[0]
audio_annotation_array= tmp[1]
print(  '\n MFCC array is \n'    + str(MFCC_array)  )
print(  '\n dimension of each MFCC array in each chunk is \n' + str(MFCC_array[0,:].shape)   )
print('\n dim of MFCC_array is \n' + str(MFCC_array.shape))
print('\n dim of audio_annotation_array is  \n' + str(audio_annotation_array.shape)  )

data_dict={1:MFCC_array, 2:audio_annotation_array}
pickle_out=open('dict.pickle', 'wb')
pickle.dump(data_dict,pickle_out)
pickle_out.close()
pickle_in=open('dict.pickle', 'rb')
data_dict2=pickle.load(pickle_in)
print(data_dict2)

'''
