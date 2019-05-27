
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

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

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
from build_MFCC_and_pickle import build_MFCC_for_audio_seg_list, get_mfcc_and_annotations, build_MFCC_for_audio_seg_list_and_pickle
from build_models_and_test import perform_cross_validation_for_one_model, prediction_results, pred_after_PCA
from visualize_data import visualize_train_test_data
import pickle
import numpy as np
import scipy
import os


np.set_printoptions(precision=1000)



folder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'

file1='ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.csv'
file2='ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.csv'
file3='ConscinciasParalelasN11-OEspelhoEOReflexoFantasiasEPerplexidadesParte413-12-1994.csv'
file4='eatmycountry1609.csv'
file6='theconcert16.csv'


file1_mp3=  'ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.mp3'
file2_mp3= 'ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.mp3'
file3_mp3 ='ConscinciasParalelasN11-OEspelhoEOReflexoFantasiasEPerplexidadesParte413-12-1994.mp3'
file4_mp3='eatmycountry1609.mp3'
file6_mp3='theconcert16.mp3'



''' Below we build the test MFCC and annotations from another online dataset, file3, e.g.'''

df=df_et_matrix_one_file(folder,file4)[0] #change at the audio=os.path.join() too!
#df=df_et_matrix_one_file(folder,file+'file_no')[0] #change at the audio=os.path.join() too!
df_nonoverlapping=make_non_time_overlapping_df_from_time_overlapping_df(df)
#print("df_nonoverlapping is \n" + str(df_nonoverlapping))
df_start_endtime_annotation_chunks= make_time_chunks_from_df_of_start_end_times(df_nonoverlapping, chunk_dur=1)
audio= os.path.join(folder, file4_mp3)
#audio = AudioSegment.from_file(audio, format="wav", chennels=1)
audio_seg_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[0]
audio_annotation_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[1]
chunk_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[2]
tmp=build_MFCC_for_audio_seg_list_and_pickle(folder,chunk_list,audio_annotation_list)
#X, Y= MFCC_array, audio_annotation_array
X4 = tmp[0]
Y4 = tmp[1] #at the audio=os.path.join() too!
#visualize_train_test_data(X,Y)
#temp = perform_cross_validation_for_one_model(X2, Y2)
#model2 = tmp[2]


model_dict4 = {2:X4, 3:Y4}
pickle_out = open('model_dict4.pickle', 'wb')
pickle.dump(model_dict4, pickle_out)
pickle_out.close()
pickle_in = open('model_dict4.pickle', 'rb')
model_dict4 = pickle.load(pickle_in)

try:
    foo_4 = pickle.load(open("model_dict4.pickle", "rb"))
except (OSError, IOError) as e:
    foo_4 = 9
    pickle.dump(foo, open("model_dict4.pickle", "wb"))

X4 = foo_4[2]
Y4 = foo_4[3]

perform_cross_validation_for_one_model(X4,Y4)