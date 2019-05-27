
'''
Visualize certain MFCC features below

'''

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from convert_time_overlapping_df_into_time_non_overapping_df import make_non_time_overlapping_df_from_time_overlapping_df
from slice_sound import make_sound_chunks_from_df_start_endtime_annotation_chunks, make_time_chunks_from_df_of_start_end_times
import os
from make_df_et_matrix import df_et_matrix_one_file
from slice_sound import make_sound_chunks_from_df_start_endtime_annotation_chunks
import pickle

def visualize_train_test_data(X,Y):
    '''
    :param X: np array of original feature values, i.e. MFCC_array
    '''
    MFCC_feats = ['f' + str(i) for i in range(0, X.shape[1])]
    #dat_MFCC = pd.DataFrame(X, columns = MFCC_feats)
    df_MFCC = pd.DataFrame(X, columns = MFCC_feats)
    df_annotations = pd.DataFrame(Y, columns=['annotations'])
    df = pd.concat((df_MFCC, df_annotations), axis=1)
    sns.barplot(x='annotations', y='f30', data=df)
    plt.show()
    sns.barplot(x='annotations', y='f100', data=df)
    plt.show()
    sns.barplot(x='annotations', y='f150', data=df)
    plt.show()

'''
#Test the fn above:

#folder="/home/susovan/Documents/music_detection/muspeak_mirex2015/music_chunks"
#sound_slice="sound-174.wav"
folder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'
file1='ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.csv'
file1_mp3=  'ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.mp3'
file2 = 'ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.csv'
file2_mp3= 'ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.mp3'


#Below, choose the filex carefully, and remember to modify the file_mp3 name to file_mp3x as well!
df=df_et_matrix_one_file(folder,file2)[0]
df_nonoverlapping=make_non_time_overlapping_df_from_time_overlapping_df(df)
#print("\n df_nonoverlapping is \n" + str(df_nonoverlapping))
df_start_endtime_annotation_chunks= make_time_chunks_from_df_of_start_end_times(df_nonoverlapping, chunk_dur=10)
audio= os.path.join(folder, file2_mp3)
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


visualize_data(X,Y)

'''
