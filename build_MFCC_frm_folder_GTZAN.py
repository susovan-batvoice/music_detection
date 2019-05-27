
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
import pickle
import numpy as np
import scipy
import os

np.set_printoptions(precision=1000)


def divide_intvl_into_subtnvl_of_given_size(ini_pt, fin_pt, len_subintvl):
    l=[]
    while ini_pt <= fin_pt:
        l.append(ini_pt)
        ini_pt=ini_pt+len_subintvl
    l=np.array(l)
    return l

def music_MFCC_frm_music_folder(dir_read_music, dir_export_music):

    music_files = []  # stores the names of all full_filename 's
    music_annotation_list = []
    ctr = -1
    music_chunk_list = []
    for r, d, f in os.walk(dir_read_music):  # root, dir, file, still not clear about the efficacy of os.walk
        for file in f:
            if '.wav' in file:
                # below, we make a list of the annotated files
                full_filename = os.path.join(r, file)
                music_files.append(full_filename)
                

    ct_list = []
    l_list = []
    for j in range(len(music_files)):
        full_filename = music_files[j]
        audio = AudioSegment.from_file(full_filename, format="wav").set_channels(1)
        l = divide_intvl_into_subtnvl_of_given_size(ini_pt=0, fin_pt=len(audio), len_subintvl=1000)
        l_list.append(l)
        for ct in range(len(l) - 1):
            ct_list.append(ct)
            audio_seg = audio[l[ct]:l[ct + 1]]
            ctr = ctr + 1
            audio_seg_handle = audio_seg.export(os.path.join(dir_export_music, "sound-%s.wav" % ctr), format="wav",
                                                parameters=["-ac", "1"])
            music_chunk_list.append(os.path.join(dir_export_music, "sound-%s.wav" % ctr))
            music_annotation_list.append('m')

    music_MFCC_array, music_annotation_array \
                = build_MFCC_for_audio_seg_list_and_pickle(dir_export_music, music_chunk_list, music_annotation_list)
    return music_MFCC_array, music_annotation_array

'''
#Test the above function:

dir_read_music = '/home/susovan/Documents/music_detection/muspeak_mirex2015/music_speech_GTZAN/music_wav'
dir_export_music = '/home/susovan/Documents/music_detection/muspeak_mirex2015/music_speech_GTZAN/chunk_GTZAN_music'

music_MFCC_array, music_annotation_array = music_MFCC_frm_music_folder(dir_read_music, dir_export_music)




print(' \n music MFCC is \n' + str(music_MFCC_array))
print(music_MFCC_array.shape)
music_dict_GTZAN = {0:music_MFCC_array, 1:music_annotation_array}
pickle_out = open('music_dict_GTZAN.pickle', 'wb')
pickle.dump(music_dict_GTZAN, pickle_out)
pickle_out.close()
pickle_in = open('music_dict_GTZAN.pickle', 'rb')
music_dict_GTZAN2 = pickle.load(pickle_in)

'''



def speech_MFCC_frm_speech_folder(dir_read_speech, dir_export_speech):
    '''

    :param dir_read_speech: str, reads the speech files from this dir, if it contains only speeches
    :param dir_export_speech: str, after pydub, we export the sppech seg here
    :return: MFCC array, speech annotation array,
    '''

    speech_files = []  # stores the names of all full_filename 's
    speech_annotation_list = []
    ctr = -1
    speech_chunk_list = []
    for r, d, f in os.walk(dir_read_speech):  # root, dir, file, still not clear about the efficacy of os.walk
        for file in f:
            if '.wav' in file:
                # below, we make a list of the annotated files
                full_filename = os.path.join(r, file)
                speech_files.append(full_filename)

    ct_list = []
    l_list = []
    for j in range(len(speech_files)):
        full_filename = speech_files[j]
        audio = AudioSegment.from_file(full_filename, format="wav").set_channels(1)
        l = divide_intvl_into_subtnvl_of_given_size(ini_pt=0, fin_pt=len(audio), len_subintvl=1000)
        l_list.append(l)
        for ct in range(len(l) - 1):
            ct_list.append(ct)
            audio_seg = audio[l[ct]:l[ct + 1]]
            ctr = ctr + 1
            audio_seg_handle = audio_seg.export(os.path.join(dir_export_speech, "sound-%s.wav" % ctr), format="wav",
                                                parameters=["-ac", "1"])
            speech_chunk_list.append(os.path.join(dir_export_speech, "sound-%s.wav" % ctr))
            speech_annotation_list.append('s')

    speech_MFCC_array, speech_annotation_array \
                = build_MFCC_for_audio_seg_list_and_pickle(dir_export_speech, speech_chunk_list, speech_annotation_list)

    return speech_MFCC_array, speech_annotation_array

'''
#Test the above function to build speech_MFCC_array and annotations
dir_read_speech= '/home/susovan/Documents/music_detection/muspeak_mirex2015/music_speech_GTZAN/speech_wav'
dir_export_speech ='/home/susovan/Documents/music_detection/muspeak_mirex2015/music_speech_GTZAN/chunk_GTZAN_speech'

speech_MFCC_array, speech_annotation_array = speech_MFCC_frm_speech_folder(dir_read_speech, dir_export_speech)

#Test the above function
print(' \n speech MFCC is \n' + str(speech_MFCC_array))
print(speech_MFCC_array.shape)
speech_dict_GTZAN = {0:speech_MFCC_array, 1:speech_annotation_array}
pickle_out = open('speech_dict_GTZAN.pickle', 'wb')
pickle.dump(speech_dict_GTZAN, pickle_out)
pickle_out.close()
pickle_in = open('speech_dict_GTZAN.pickle', 'rb')
speech_dict_GTZAN2 = pickle.load(pickle_in)
'''
#load the music data
try:
    foo_mus_GTZAN = pickle.load(open("music_dict_GTZAN.pickle", "rb"))
except (OSError, IOError) as e:
    foo_mus_GTZAN = 3
    pickle.dump(foo_mus_GTZAN, open("music_dict_GTZAN.pickle", "wb"))
    pickle.dump(foo_mus_GTZAN, open("music_dict_GTZAN.pickle", "wb"))

X_music_GTZAN = foo_mus_GTZAN[0]
Y_music_GTZAN= foo_mus_GTZAN[1]#load the music data




#load the speech data

try:
    foo_sp_GTZAN = pickle.load(open("speech_dict_GTZAN.pickle", "rb"))
except (OSError, IOError) as e:
    foo_sp_GTZAN = 3
    pickle.dump(foo_sp_GTZAN, open("speech_dict_GTZAN.pickle", "wb"))
    pickle.dump(foo_sp_GTZAN, open("speech_dict_GTZAN.pickle", "wb"))



X_speech_GTZAN = foo_sp_GTZAN[0]
Y_speech_GTZAN= foo_sp_GTZAN[1]





X_GTZAN = np.concatenate( (X_music_GTZAN, X_speech_GTZAN), axis=0)
Y_GTZAN = np.concatenate( (Y_music_GTZAN, Y_speech_GTZAN), axis=0)
model_GTZAN = perform_cross_validation_for_one_model(X_GTZAN, Y_GTZAN)[2]

'''Below, we define the prediction with the above model'''

def prediction_results1(model, X_train, Y_train, X_test, Y_test):
    '''
    :param model: a model that's not fitted yet, eg. LR
    :param X_train: np array, training MFCC to fit the model BEFORE PCA
    :param Y_train: np array, training annotations to for the model
    :param X_test: np array, test MFCC, ideally from a different data from where X_train came
    :param Y_test: np array, test annotations, ideally from a different data from where Y_train came
    :return: prediction results
    GOAL: is to see how our model performs when trained on one data and then getting tested on another
    '''

    #checking for class imbalane below
    print('\n in the training data, the ratio of music to speech is ' + str(Y_train[Y_train == 'm'].shape[0] / Y_train[Y_train == 's'].shape[0]))
    print('\n in the test data, the ratio of music to speech is ' + str(Y_test[Y_test == 'm'].shape[0] / Y_test[Y_test == 's'].shape[0]))
    #feature scaling
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    #fitting the model below
    #X_train, X_test = scale_data(X_train, X_test)
    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    print('\n test accuracy for ' + str(model) + 'before PCA and hyp. opt. is ' + str(
       accuracy_score(Y_test, predictions)))
    print('\n confusion matrix for ' + str(model) + 'before PCA and hyp. opt. is \n' + str(
        confusion_matrix(Y_test, predictions)))
    print('\n detailed classification results for test data, before PCA and hyp. opt. are \n' + str(
       classification_report(Y_test, predictions)))





def pred_after_PCA1(model, X_train, Y_train, X_test, Y_test):
    '''

    :param model: a model that's not fitted yet, eg. LR
    :param X_train: np array, training MFCC to fit the model after PCA
    :param Y_train: np array, training annotations to for the model
    :param X_test: np array, test MFCC, ideally from a different data from where X_train came
    :param Y_test: np array, test annotations, ideally from a different data from where Y_train came
    :return: prediction results
    GOAL: is to see how our model performs when trained on one data and then getting tested on another
    '''
    print('\n in the training data, the ratio of music to speech is ' + str(Y_train[Y_train == 'm'].shape[0] / Y_train[Y_train == 's'].shape[0]))
    print('\n in the test data, the ratio of music to speech is ' + str(Y_test[Y_test == 'm'].shape[0] / Y_test[Y_test == 's'].shape[0]))
    print('PERFORMING PCA \n')
    pca = PCA(n_components=150)
    #pca = PCA(.95)
    #X_train, X_test = scale_data(X_train, X_test)
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    pca.fit(X_train)
    X_train_PCA = pca.transform(X_train)
    X_test_PCA = pca.transform(X_test)
    model.fit(X_train_PCA, Y_train)
    pred_test_PCA = model.predict(X_test_PCA)
    #print('\n training accuracy for' +  str(model)  +  'after PCA is '  +  str(accuracy_score(Y_train, pred_train_PCA)) )
    print('\n test accuracy for ' + str(model) + 'after PCA is ' + str(accuracy_score(Y_test, pred_test_PCA)))
    predictions = model.predict(X_test_PCA)
    #print('\n test accuracy for ' + str(model) + 'after PCA  is '  +  str (accuracy_score(Y_test, predictions))  )
    print('\n confusion matrix for ' + str(model) + 'after PCA is \n' + str(confusion_matrix(Y_test, predictions)) )
    print('\n detailed classification results for test data, after PCA are \n ' + str(classification_report(Y_test, predictions)) )


'''perform load the test data (CW) below'''

try:
    foo_0 = pickle.load(open("model_dict0.pickle", "rb"))
except (OSError, IOError) as e:
    foo_0 = 1
    pickle.dump(foo, open("model_dict0.pickle", "wb"))
X0 = foo_0[2]
Y0 = foo_0[3]
where_music = np.where(Y0 == 'm')[0] #array tells s where the segment is music for file 1 segments
where_speech = np.where(Y0 == 's')[0]
#where_music_for_model = where_music[0: int(0.32 * len(where_speech))] #as m to s ratio in cw data is 0.32,
where_speech_for_model = where_speech[0: len(where_music)]
X0_music = X0[where_music]
Y0_music = Y0[where_music]
X0_speech = X0[where_speech_for_model]
Y0_speech = Y0[where_speech_for_model]



'''Performing grid search below'''
prediction_results1(model_GTZAN, X_GTZAN, Y_GTZAN, X0, Y0)




#Below, we perturb the above model-GTZAN's parameters, and run each for prediction purposes
parameters = {'kernel':('rbf', 'linear'), 'C':(10, 5, 1),'gamma': (0.004,'auto', 0.006)}
#clf_GS = GridSearchCV(estimator=model_GTZAN, param_grid=parameters, scoring='accuracy', cv=5)
#clf_GS.fit(X_GTZAN,Y_GTZAN)
#best_accuracy = clf_GS.best_score_
#best_params = clf_GS.best_params_
#print("The best parameters for accuracy before PCA are: \n" + str (best_params) )
l = list(parameters.values())
C_list=[]
gamma_list=[]

    for j in range(len(l[i])):
        C = l[i][j]
        gamma= l[2][j]
        model_GTZAN.C = C
        model_GTZAN.gamma = gamma






print( 'test accuracy for hyper parameter optimized SVM, before PCA is \n'  +  str (accuracy_score(Y0, predictions_opt))  )
print('confusion matrix for SVM on test data, before PCA is \n' + str(confusion_matrix(Y0, predictions_opt)) )
print('detailed classification results for test data, before PCA are \n' + str(classification_report(Y0, predictions_opt)) )