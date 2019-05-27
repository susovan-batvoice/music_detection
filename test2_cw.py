
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
from build_MFCC_and_pickle import get_mfcc_and_annotations, build_MFCC_for_audio_seg_list_and_pickle
from build_models_and_test import perform_cross_validation_for_one_model, prediction_results, pred_after_PCA
import pickle
import numpy as np
import scipy
import os





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

X0_new = np.concatenate((X0_music, X0_speech), axis=0)
Y0_new = np.concatenate((Y0_music, Y0_speech), axis=0)

model0 = perform_cross_validation_for_one_model(X0, Y0)[2]

#below, we try to find al already built model and the data from fime 1 (online data)
try:
    foo_1 = pickle.load(open("model_dict1.pickle", "rb"))
except (OSError, IOError) as e:
    foo_1 = 3
    pickle.dump(foo, open("model_dict1.pickle", "wb"))
    pickle.dump(foo, open("model_dict.pickle", "wb"))

#model1 = foo_1[1]
''' Below, we do a small experiment to train the classifier so that it purpusefully reflects the class imbl. between
m and s like in cw, so that m is 40%, s is 70%'''
X1 = foo_1[2]
Y1 = foo_1[3]
where_music = np.where(Y1 == 'm')[0] #array tells s where the segment is music for file 1 segments
where_speech = np.where(Y1 == 's')[0]
where_music_for_model = where_music[0: int(0.32 * len(where_speech))] #as m to s ratio in cw data is 0.32,
X1_music = X1[where_music_for_model]
Y1_music = Y1[where_music_for_model]
X1_speech = X1[where_speech]
Y1_speech = Y1[where_speech]

X1_new = np.concatenate((X1_music, X1_speech), axis=0)
Y1_new = np.concatenate((Y1_music, Y1_speech), axis=0)

model1_new = perform_cross_validation_for_one_model(X1, Y1)[2]
model1 = perform_cross_validation_for_one_model(X1, Y1)[2]

#below, we try to find al already built model and the data from fime 2 (online data)
try:
    foo_2 = pickle.load(open("model_dict2.pickle", "rb"))
except (OSError, IOError) as e:
    foo_2 = 5
    pickle.dump(foo, open("model_dict2.pickle", "wb"))

#model1 = foo_2[1]
X2 = foo_2[2]
Y2 = foo_2[3]


where_music = np.where(Y2 == 'm')[0] #array tells s where the segment is music for file 1 segments
where_speech = np.where(Y2 == 's')[0]
where_music_for_model = where_music[0: int(0.32 * len(where_speech))] #as m to s ratio in cw data is 0.32,
X2_music = X2[where_music_for_model]
Y2_music = Y2[where_music_for_model]
X2_speech = X2[where_speech]
Y2_speech = Y2[where_speech]

X2_new = np.concatenate((X1_music, X1_speech), axis=0)
Y2_new = np.concatenate((Y1_music, Y1_speech), axis=0)

#model2_new = perform_cross_validation_for_one_model(X2, Y2)[2]
model2 = perform_cross_validation_for_one_model(X2, Y2)[2]


try:
    foo_3 = pickle.load(open("model_dict3.pickle", "rb"))
except (OSError, IOError) as e:
    foo_3 = 7
    pickle.dump(foo, open("model_dict3.pickle", "wb"))

X3 = foo_3[2]
Y3 = foo_3[3]

where_music = np.where(Y3 == 'm')[0] #array tells s where the segment is music for file 1 segments
where_speech = np.where(Y3 == 's')[0]
where_music_for_model = where_music[0: int(0.32 * len(where_speech))] #as m to s ratio in cw data is 0.32,
X3_music = X3[where_music_for_model]
Y3_music = Y3[where_music_for_model]
X3_speech = X3[where_speech]
Y3_speech = Y3[where_speech]

X3_new = np.concatenate((X3_music, X3_speech), axis=0)
Y3_new = np.concatenate((Y3_music, Y3_speech), axis=0)

model3_new = perform_cross_validation_for_one_model(X3, Y3)[2]
model3 = perform_cross_validation_for_one_model(X3, Y3)[2]



try:
    foo_4 = pickle.load(open("model_dict4.pickle", "rb"))
except (OSError, IOError) as e:
    foo_4 = 9
    pickle.dump(foo, open("model_dict4.pickle", "wb"))

X4 = foo_4[2]
Y4 = foo_4[3]
where_music = np.where(Y4 == 'm')[0] #array tells s where the segment is music for file 1 segments
where_speech = np.where(Y4 == 's')[0]
where_music_for_model = where_music[0: int(0.32 * len(where_speech))] #as m to s ratio in cw data is 0.32,
X4_music = X4[where_music_for_model]
Y4_music = Y4[where_music_for_model]
X4_speech = X4[where_speech]
Y4_speech = Y4[where_speech]

X4_new = np.concatenate((X4_music, X4_speech), axis=0)
Y4_new = np.concatenate((Y4_music, Y4_speech), axis=0)

model4_new = perform_cross_validation_for_one_model(X4, Y4)[2]
model4 = perform_cross_validation_for_one_model(X4, Y4)[2]


X = np.concatenate((X1, X2, X3, X4), axis=0)
Y =  np.concatenate((Y1, Y2, Y3, Y4), axis=0)

where_music = np.where(Y == 'm')[0] #array tells s where the segment is music for file 1 segments
where_speech = np.where(Y == 's')[0]
where_music_for_model = where_music[0: int(0.32 * len(where_speech))] #as m to s ratio in cw data is 0.32,
X_music = X[where_music_for_model]
Y_music = Y[where_music_for_model]
X_speech = X[where_speech]
Y_speech = Y[where_speech]

X_new = np.concatenate((X_music, X_speech), axis=0)
Y_new = np.concatenate((Y_music, Y_speech), axis=0)

model_new = perform_cross_validation_for_one_model(X, Y)[2]




model = perform_cross_validation_for_one_model(X,Y)[2]


def scale_data(X_train, X_test):
    '''
    scaling the test data according to the training data
    :param X_train: np array, training data
    :param Y_train: np array,
    :param X_test:
    :param Y_test:
    :return:
    '''
    mean_train = np.mean(X1, axis=0)
    std_train = np.std(X1, axis=0)
    X_train = (X_train - mean_train) / std_train
    X_test = (X_test - mean_train) / std_train
    return X_train, X_test

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

print("PREDICTIONS BEFORE PCA")

#prediction_results1(model1, X1_new, Y1_new, X0, Y0)
#prediction_results1(model1, X1, Y1, X0, Y0)



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
    pca = PCA(n_components=21)
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




def prediction_results2(model, X_train, Y_train, X_test, Y_test):
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
    #X_train = preprocessing.scale(X_train)
    #X_test = preprocessing.scale(X_test)
    #fitting the model below
    X_train, X_test = scale_data(X_train, X_test)
    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    print('\n test accuracy for ' + str(model) + 'before PCA and hyp. opt. is ' + str(
       accuracy_score(Y_test, predictions)))
    print('\n confusion matrix for ' + str(model) + 'before PCA and hyp. opt. is \n' + str(
        confusion_matrix(Y_test, predictions)))
    print('\n detailed classification results for test data, before PCA and hyp. opt. are \n' + str(
       classification_report(Y_test, predictions)))

print("PREDICTIONS BEFORE PCA")

prediction_results2(model1, X1_new, Y1_new, X0, Y0)
prediction_results2(model1, X1, Y1, X0, Y0)

def pred_after_PCA2(model, X_train, Y_train, X_test, Y_test):
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
    pca = PCA(n_components=21)
    #pca = PCA(.95)
    X_train, X_test = scale_data(X_train, X_test)
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

print("PREDICTIONS AFTER PCA")
#pred_after_PCA2(model, X, Y,  MFCC_array, audio_annotation_array)
#pred_after_PCA2(model1, X1_new, Y1_new, X0, Y0)
#pred_after_PCA2(model1, X1, Y1, X0_new, Y0_new)
#pred_after_PCA2(model3_new, X3_new, Y3_new, X0, Y0)
#pred_after_PCA2(model_new, X_new, Y_new, X0, Y0)
pred_after_PCA2(model2, X2, Y2, X0, Y0)