
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pydub import AudioSegment
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd
from convert_time_overlapping_df_into_time_non_overapping_df import make_non_time_overlapping_df_from_time_overlapping_df
import os
from make_df_et_matrix import df_et_matrix_one_file


def divide_intvl_into_subtnvl_of_given_size(ini_pt, fin_pt, len_subintvl):
    l=[]
    while ini_pt <= fin_pt:
        l.append(ini_pt)
        ini_pt=ini_pt+len_subintvl
    l=np.array(l)
    return l


'''
def chunk_sound(folder,mp3_file, chunk_dur,\
                dir_name="/home/susovan/Documents/music_detection/muspeak_mirex2015/music_chunks"):
    

    :param folder: str
    :param mp3_file: .mp3 file
    :param chunk_dur: scalar, the chunk duration to segment the audio
    :return: ???
    
    full_name_mp3file=os.path.join(folder,mp3_file)
    sound = AudioSegment.from_file(full_name_mp3file, format="mp3")
    #sound=sound[:20*1000]
    "If the wavfile is too large, there can be a memory issue, raising the error" \
    " pydub.exceptions.CouldntDecodeError: Couldn't find data header in wav data"
    #dir_name="/home/susovan/Documents/music_detection/muspeak_mirex2015/music_chunks"
    for i, chunk in enumerate(sound[::chunk_dur*1000]):
        with open(os.path.join(dir_name,"sound-%s.mp3" % i) , "wb") as f:
            chunk.export(f, format="mp3")
    return chunk
    
    '''


def make_time_chunks_from_df_of_start_end_times(df_nonoverlapping, chunk_dur):
    '''
    Below:
    df=df_et_matrix_one_file(folder,file1)[0]
    df_nonoverlapping=make_non_time_overlapping_df_from_time_overlapping_df(df)

    :param df: pd dataframe containing the array of start and end times [S,E] of music and speech with annotations (m, s)
    chunk_dur is in seconds,
    :return: list of all chunks of of a certain chunk duration chunk_dur for each S and E
    lst is the list of all chunks of duration chunk_dur if the audio seglent is at least chun_dur sec. long
    lst2 is the list of all corresponding annotations (music or speech)
    '''

    x = df_nonoverlapping.values[:, 0:2]# x is the matrix of start and endtimes so that music and speech don't overlap
    x = x.astype(float)
    y = df_nonoverlapping.values[:, 2]  # y is the corresponding array of the corresponding annotations (m, s)
    #y=np.array([y]).T #to make y vertical
    lst=[] #list of np arrays of starttimes and endtimes of length chunk_dur, note\
    # that x.shape[0] is the number of annotated audio segments in the initial audio file
    lst2= [] #stores the annotations 'm or s) for the above
    count_chunk_list=[]
    for i in range(x.shape[0]):#for i-th row of x
        if x[i,1]-x[i,0] >= chunk_dur:
            #ct_chunks=int((x[i,1]-x[i,0])/chunk_dur) #counts number of chunks
            #count_chunk_list.append(ct_chunks)
            #ct_chunks = round((x[i, 1] - x[i, 0]) / chunk_dur)  # counts number of chunks
            #lst.append(  np.linspace( x[i,0], x[i,1], ct_chunks )   )
            lst.append(divide_intvl_into_subtnvl_of_given_size(ini_pt=x[i,0], fin_pt=x[i,1], len_subintvl=chunk_dur))
            tmp = y[i]
            #lst2[i].append( tmp for k in range(ct_chunks))
            ct_chunks= len( divide_intvl_into_subtnvl_of_given_size(ini_pt=x[i,0], fin_pt=x[i,1], len_subintvl=chunk_dur) )
            lst2.append( [tmp for k in range(ct_chunks)]  )

    list_start_endtime_for_chunks= []
    list_annotation_for_chunks=[]
    #df_start_endtime_chunks=pd.DataFrame({''})
    for i in range(len(lst)):
        if lst[i]!= []:
           list_start_endtime_for_chunks.append(lst[i])
           list_annotation_for_chunks.append(lst2[i])
    df_start_endtime_annotation_chunks = pd.DataFrame({'list_start_endtime_for_chunks': list_start_endtime_for_chunks, 'list_annotation_for_chunks': list_annotation_for_chunks})
    return df_start_endtime_annotation_chunks

def make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio, \
                                                              dir_name = "/home/susovan/Documents/music_detection/muspeak_mirex2015/music_chunks"):
    '''


    :param df_start_endtime_annotation_chunks: defined above, it's a df that contains 1) list a of np arrays that stores
    the vectors of chunk_dur long sound chunk from an audio file, and 2) list of corr. annotations, m or s.
    audio: initial audio file, ideally would have 1 channel
    :param dir_name: str, name of dir where audio files will be saved
    :return: corresponding df of audio segments (using pydub) and corresponding annotations
    '''

    audio = AudioSegment.from_file(audio, format="mp3").set_channels(1)
    df_col_names=df_start_endtime_annotation_chunks.columns.values #gives the names of the columns
    list_start_endtime_for_chunks=df_start_endtime_annotation_chunks[df_col_names[0]].tolist()
    list_annotation_for_chunks=df_start_endtime_annotation_chunks[df_col_names[1]].tolist()
    audio_seg_lst=[] #contains the audio seg objects, without being exported
    audio_annotation_lst=[]
    chunk_list = []  #contains the corrsponding audio segments in audio seg_list after being exported as .mp3/wav
    ctr=-1
    for i in range(len(list_start_endtime_for_chunks)): #for i th array annotated time intervals of the original file
        for j in range(len(list_start_endtime_for_chunks[i])-1): #for j th time chunk of the the i -th annotated time interval
            start_tm_chunk_in_ms = list_start_endtime_for_chunks[i][j]*1000
            end_tm_chunk_in_ms = list_start_endtime_for_chunks[i][j+1]*1000 #times in millisec.
            audio_seg = audio[start_tm_chunk_in_ms:end_tm_chunk_in_ms]
            audio_seg_lst.append(audio_seg)
            audio_annotation_lst.append(list_annotation_for_chunks[i][j])
            #audio_seg_handle = audio_seg.export(os.path.join(dir_name,"sound-%s.mp3" % j), format="mp3")
            ctr = ctr + 1
            audio_seg_handle = audio_seg.export(os.path.join(dir_name, "sound-%s.wav" % ctr), format="wav", parameters=["-ac", "1"])
            chunk_list.append(os.path.join(dir_name,"sound-%s.wav" % ctr))


    return audio_seg_lst, audio_annotation_lst, chunk_list






'''
#TEST the above fn:
folder="/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples"
#wav_file="ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.mp3.wav"
#mp3_file="ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.mp3"
mp3_file="ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.mp3"
chunk_dur=10
chunk=chunk_sound(folder,mp3_file,chunk_dur,dir_name=folder)
print( "\n chunks are  \n")

'''

'''

#TEST the fn make_time_chunks_from_df_of_start_end_times:

folder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'
file1='/ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.csv'
df=df_et_matrix_one_file(folder,file1)[0]
df_nonoverlapping=make_non_time_overlapping_df_from_time_overlapping_df(df)
tmp2=make_time_chunks_from_df_of_start_end_times(df_nonoverlapping, chunk_dur=10)
print( '\n  The df containing segments of chunk_dur and annotations are below   \n' + str(tmp2) )

'''


'''
#TEST make_sound_chunks_from_df_start_endtime_annotation_chunks

folder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'
#file1='ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.csv'
#file1_mp3=  '/ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.mp3'


file2='ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.csv'
file2_mp3='ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.mp3'


#file3='ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.csv'
#file3_mp3='ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.mp3'


df=df_et_matrix_one_file(folder,file2)[0]
df_nonoverlapping=make_non_time_overlapping_df_from_time_overlapping_df(df)
print("df_nonoverlapping is \n" + str(df_nonoverlapping))
df_start_endtime_annotation_chunks= make_time_chunks_from_df_of_start_end_times(df_nonoverlapping, chunk_dur=10)

audio= os.path.join(folder, file2_mp3)
#audio= folder + file2_mp3
#audio = AudioSegment.from_file(audio, format="mp3", chennels=1)
#audio = AudioSegment.from_file(audio, format="wav", chennels=1)
audio_seg_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[0]
audio_annotation_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[1]
chunk_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[2]
print(  "\n df_start_endtime_annotation_chunks is \n" + str(df_start_endtime_annotation_chunks) )
print( "\n audio_seg_list \n"    + str(audio_seg_list) )
print( "\n audio_annotation_list \n" + str(audio_annotation_list) )
print("\n chunk_list is \n" + str(chunk_list) )

'''