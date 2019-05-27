
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
from build_MFCC_and_pickle import build_MFCC_for_audio_seg_list, get_mfcc_and_annotations, build_MFCC_for_audio_seg_list_and_pickle
from build_models_and_test import perform_cross_validation_for_one_model, prediction_results, pred_after_PCA
from visualize_data import visualize_train_test_data
import pickle
import numpy as np
import scipy
import os


def divide_intvl_into_subtnvl_of_given_size(ini_pt, fin_pt, len_subintvl):
    l=[]
    while ini_pt <= fin_pt:
        l.append(ini_pt)
        ini_pt=ini_pt+len_subintvl
    l=np.array(l)
    return l



np.set_printoptions(precision=1000)

'''First we resample matrix to use for MFCC, in case the lengths are different'''

def resample_matrix(F,num):
    G=np.empty((F.shape[0], num))
    for i in range(F.shape[0]):
      G[i]=scipy.signal.resample(F[i], num, t=None, axis=0, window=None)
    return G

folder = '/home/susovan/Documents/music_detection/muspeak_mirex2015/labeled_seg_callwatch' #folder name containing the .wav files of callwatch
dir_name = '/home/susovan/Documents/music_detection/muspeak_mirex2015/chunked_segments_callwatch'

'''Below, we chunk the test files (callwatch data) into smaller chunks'''





'''Below, we extract features of test data using Py Audio Analysis, resample all feature vectors, and 
store them in the list feat_list'''
files=[] #stores the names of all full_filename 's
Fs_lst=[]
x_lst=[]
MFCC_list=[]
audio_annotation_list=[]
ctr=-1
chunk_list=[]
for r, d, f in os.walk(folder): #root, dir, file, still not clear about the efficacy of os.walk
    for file in f:
        if '.wav' in file:
            #below, we make a list of the annotated files
            full_filename=os.path.join(r, file)
            files.append(full_filename)
ct_list=[]
l_list=[]
for j in range(len(files)):
    full_filename=files[j]
    audio = AudioSegment.from_file(full_filename, format="wav").set_channels(1)
    l = divide_intvl_into_subtnvl_of_given_size(ini_pt=0, fin_pt=len(audio), len_subintvl=1000)
    l_list.append(l)
    for ct in range(len(l)-1):
        ct_list.append(ct)
        #ctr = ctr + 1
        audio_seg = audio[l[ct]:l[ct+1]]
        ctr = ctr + 1
        if 'speech' in full_filename:
            audio_annotation_list.append('s')
        else:
            audio_annotation_list.append('m')
        audio_seg_handle = audio_seg.export(os.path.join(dir_name, "sound-%s.wav" % ctr), format="wav", parameters=["-ac", "1"])
        chunk_list.append(os.path.join(dir_name, "sound-%s.wav" % ctr))


MFCC_array, audio_annotation_array = build_MFCC_for_audio_seg_list_and_pickle(folder,chunk_list, audio_annotation_list)
'''
for full_filename in files:
#Below we compute the MFCC
    [Fs, x] = audioBasicIO.readAudioFile(full_filename)
    Fs_lst.append(Fs)
    x_lst.append(x)
    F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
    G=resample_matrix(F,num=15) #see the definitionog resample_matrix above
    #feat_list.append(G)
    MFCC = G[8:21, :]  # 9th to 21-st features are the MFCC coeffs
    MFCC_flat = np.ndarray.flatten(MFCC)  # flatening the array, but are we destroying time series structure?
    MFCC_list.append(MFCC_flat)


MFCC_array = np.asarray(MFCC_list)
audio_annotation_array=np.asarray(audio_annotation_list)

'''
print(MFCC_array)
print(MFCC_array.shape)
print(audio_annotation_array)
print(audio_annotation_array.shape)


#below, we try to find al already built model
try:
    foo_1 = pickle.load(open("model_dict1.pickle", "rb"))
except (OSError, IOError) as e:
    foo_1 = 3
    pickle.dump(foo, open("model_dict1.pickle", "wb"))
    pickle.dump(foo, open("model_dict.pickle", "wb"))

model1 = foo_1[1]
X1 = foo_1[2]
Y1 = foo_1[3]





try:
    foo_3 = pickle.load(open("model_dict3.pickle", "rb"))
except (OSError, IOError) as e:
    foo_3 = 7
    pickle.dump(foo, open("model_dict3.pickle", "wb"))

X3 = foo_3[2]
Y3 = foo_3[3]

def prediction_results(model, X_train, Y_train, X_test, Y_test):
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
    #fitting the model below
    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    print('\n test accuracy for ' + str(model) + 'before PCA and hyp. opt. is ' + str(
       accuracy_score(Y_test, predictions)))
    print('\n confusion matrix for ' + str(model) + 'before PCA and hyp. opt. is \n' + str(
        confusion_matrix(Y_test, predictions)))
    print('\n detailed classification results for test data, before PCA and hyp. opt. are \n' + str(
       classification_report(Y_test, predictions)))

print("PEDICTIONS BEFORE PCA")
prediction_results(model1, X1, Y1, MFCC_array, audio_annotation_array)


def pred_after_PCA(model, X_train, Y_train, X_test, Y_test):
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
    pca = PCA(n_components=1)
    #pca = PCA(.95)
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


try:
    foo_2 = pickle.load(open("model_dict2.pickle", "rb"))
except (OSError, IOError) as e:
    foo_2 = 5
    pickle.dump(foo, open("model_dict2.pickle", "wb"))

model1 = foo_2[1]
X2 = foo_2[2]
Y2 = foo_2[3]



''' Below we build the test MFCC and annotations from another online dataset, file2, e.g.'''

df=df_et_matrix_one_file(folder,file3)[0] #change at the audio=os.path.join() too!
#df=df_et_matrix_one_file(folder,file+'file_no')[0] #change at the audio=os.path.join() too!
df_nonoverlapping=make_non_time_overlapping_df_from_time_overlapping_df(df)
#print("df_nonoverlapping is \n" + str(df_nonoverlapping))
df_start_endtime_annotation_chunks= make_time_chunks_from_df_of_start_end_times(df_nonoverlapping, chunk_dur=1)
audio= os.path.join(folder, file3_mp3)
#audio = AudioSegment.from_file(audio, format="wav", chennels=1)
audio_seg_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[0]
audio_annotation_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[1]
chunk_list=make_sound_chunks_from_df_start_endtime_annotation_chunks(df_start_endtime_annotation_chunks, audio)[2]
tmp=build_MFCC_for_audio_seg_list_and_pickle(folder,chunk_list,audio_annotation_list)
#X, Y= MFCC_array, audio_annotation_array
X3 = tmp[0]
Y3 = tmp[1] #at the audio=os.path.join() too!
#visualize_train_test_data(X,Y)
#temp = perform_cross_validation_for_one_model(X2, Y2)
#model2 = tmp[2]


model_dict3 = {1:model1, 2:X3, 3:Y3}
pickle_out = open('model_dict3.pickle', 'wb')
pickle.dump(model_dict3, pickle_out)
pickle_out.close()
pickle_in = open('model_dict3.pickle', 'rb')
model_dict3 = pickle.load(pickle_in)

try:
    foo_3 = pickle.load(open("model_dict3.pickle", "rb"))
except (OSError, IOError) as e:
    foo_3 = 7
    pickle.dump(foo, open("model_dict3.pickle", "wb"))

X3 = foo_3[2]
Y3 = foo_3[3]


X = np.concatenate((X1,X2),axis=0)
Y =  np.concatenate((Y1,Y2),axis=0)
prediction_results(model1, X, Y, MFCC_array, audio_annotation_array)


print("PEDICTIONS AFTER PCA")
pred_after_PCA(model1, X, Y,  MFCC_array, audio_annotation_array)