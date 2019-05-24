
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pydub import AudioSegment
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
from build_MFCC_and_pickle import build_MFCC_for_part_of_audio_seg_list, get_mfcc_and_annotations, build_MFCC_for_audio_seg_list_and_pickle
from build_models_and_test import perform_cross_validation_for_one_model, prediction_results, pred_after_PCA
from visualize_data import visualize_train_test_data
from build_MFCC_frm_folder_GTZAN import prediction_results1, pred_after_PCA1
import pickle
import numpy as np
import scipy
import os

np.set_printoptions(precision=1000)

#load the music data below
try:
    foo_mus_GTZAN = pickle.load(open("music_dict_GTZAN.pickle", "rb"))
except (OSError, IOError) as e:
    foo_mus_GTZAN = 3
    pickle.dump(foo_mus_GTZAN, open("music_dict_GTZAN.pickle", "wb"))
    pickle.dump(foo_mus_GTZAN, open("music_dict_GTZAN.pickle", "wb"))

X_music_GTZAN = foo_mus_GTZAN[0]
Y_music_GTZAN= foo_mus_GTZAN[1]#load the music data




#load the speech data below

try:
    foo_sp_GTZAN = pickle.load(open("speech_dict_GTZAN.pickle", "rb"))
except (OSError, IOError) as e:
    foo_sp_GTZAN = 3
    pickle.dump(foo_sp_GTZAN, open("speech_dict_GTZAN.pickle", "wb"))
    pickle.dump(foo_sp_GTZAN, open("speech_dict_GTZAN.pickle", "wb"))



X_speech_GTZAN = foo_sp_GTZAN[0]
Y_speech_GTZAN= foo_sp_GTZAN[1]



#combine the above music and speech data and cross validate

X_GTZAN = np.concatenate( (X_music_GTZAN, X_speech_GTZAN), axis=0)
Y_GTZAN = np.concatenate( (Y_music_GTZAN, Y_speech_GTZAN), axis=0)
model_GTZAN = perform_cross_validation_for_one_model(X_GTZAN, Y_GTZAN)[2]



print('START GRID SEARCH TO OPTIMIZE HYPERPARAMETERS OF SVM, WHICH ALREADY PERFORMED WELL WITH DEFAULT PARAMETERS \n')
parameters = {'kernel': ('rbf', 'linear'), 'C':(5,6,4),'gamma': (0.005, 0.01, 0.05)}
clf_GS = GridSearchCV(estimator = model_GTZAN, param_grid=parameters, scoring='accuracy', cv=5)
clf_GS.fit(X_GTZAN,Y_GTZAN)
best_accuracy=clf_GS.best_score_
best_params=clf_GS.best_params_
best_estimator = clf_GS.best_estimator_
print("The best parameters for accuracy before PCA are: \n" + str (best_params) )


