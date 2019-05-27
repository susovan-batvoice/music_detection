from sklearn import preprocessing
from sklearn import model_selection # for command model_selection.cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
#from scipy.stats.stats import pearsonr

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from make_df_et_matrix import df_et_matrix_one_file
from convert_time_overlapping_df_into_time_non_overapping_df import make_non_time_overlapping_df_from_time_overlapping_df
from slice_sound import make_time_chunks_from_df_of_start_end_times, make_sound_chunks_from_df_start_endtime_annotation_chunks
from build_MFCC_and_pickle import build_MFCC_for_audio_seg_list, get_mfcc_and_annotations
from build_models_and_test import perform_cross_validation_for_one_model, prediction_results, pred_after_PCA
from visualize_data import visualize_train_test_data
import pickle
import os

'''

folder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'
file1='ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.csv'
file2='ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.csv'
file3='ConscinciasParalelasN11-OEspelhoEOReflexoFantasiasEPerplexidadesParte413-12-1994.csv'
file4='eatmycountry1609.csv'
file6='theconcert16.csv'
file7='callwatch_data.csv'

file1_mp3=  'ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.mp3'
file2_mp3= 'ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.mp3'
file3_mp3 ='ConscinciasParalelasN11-OEspelhoEOReflexoFantasiasEPerplexidadesParte413-12-1994.mp3'
file4_mp3='eatmycountry1609.mp3'
file6_mp3='theconcert16.mp3'


df=df_et_matrix_one_file(folder,file2)[0] #changefolder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'
file1='ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.csv'
file2='ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.csv'
file3='ConscinciasParalelasN11-OEspelhoEOReflexoFantasiasEPerplexidadesParte413-12-1994.csv'
file4='eatmycountry1609.csv'
file6='theconcert16.csv'
file7='callwatch_data.csv'

file1_mp3=  'ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.mp3'
file2_mp3= 'ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.mp3'
file3_mp3 ='ConscinciasParalelasN11-OEspelhoEOReflexoFantasiasEPerplexidadesParte413-12-1994.mp3'
file4_mp3='eatmycountry1609.mp3'
file6_mp3='theconcert16.mp3'


df=df_et_matrix_one_file(folder,file2)[0] #change at the audio=os.path.join() too!
df_nonoverlapping=make_non_time_overlapping_df_from_time_overlapping_df(df)
#print("df_nonoverlapping is \n" + str(df_nonoverlapping))
df_start_endtime_annotation_chunks= make_time_chunks_from_df_of_start_end_times(df_nonoverlapping, chunk_dur=10)
audio= os.path.join(folder, file2_mp3)
#audio= folder + file_mp3
#audio = AudioSegment.from_file(audio, format="mp3", chennels=1)
#audio = AudioSegment.from_file(audio, format="wav", chennels=1)
audio_seg_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[0]
audio_annotation_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[1]
chunk_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[2]
tmp=build_MFCC_for_audio_seg_list(folder,chunk_list,audio_annotation_list)
MFCC_array = tmp[0]
audio_annotation_array= tmp[1] at the audio=os.path.join() too!
df_nonoverlapping=make_non_time_overlapping_df_from_time_overlapping_df(df)
#print("df_nonoverlapping is \n" + str(df_nonoverlapping))
df_start_endtime_annotation_chunks= make_time_chunks_from_df_of_start_end_times(df_nonoverlapping, chunk_dur=10)
audio= os.path.join(folder, file2_mp3)
#audio= folder + file_mp3
#audio = AudioSegment.from_file(audio, format="mp3", chennels=1)
#audio = AudioSegment.from_file(audio, format="wav", chennels=1)
audio_seg_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[0]
audio_annotation_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[1]
chunk_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[2]
tmp=build_MFCC_for_audio_seg_list(folder,chunk_list,audio_annotation_list)
MFCC_array = tmp[0]
audio_annotation_array= tmp[1]

'''


#The following part needs to be run from main()
foo = get_mfcc_and_annotations()
X = foo[1]
Y = foo[2]
#X, Y= MFCC_array, audio_annotation_array
visualize_train_test_data(X,Y)
tmp = perform_cross_validation_for_one_model(X,Y)
model = tmp[2]
prediction_results(X,Y,model)
pred_after_PCA(X,Y,model)

'''
Writing the following part for temporary use, so we can save the model for future use, using pickle
'''
model_dict = {1: model}
pickle_out = open('model_dict.pickle', 'wb')
pickle.dump(model_dict, pickle_out)
pickle_out.close()
pickle_in = open('model_dict.pickle', 'rb')
model_dict2 = pickle.load(pickle_in)

try:
    foo = pickle.load(open("model_dict.pickle", "rb"))
except (OSError, IOError) as e:
    foo = 3
    pickle.dump(foo, open("model_dict.pickle", "wb"))


