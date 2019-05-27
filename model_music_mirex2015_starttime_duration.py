'''We test on the music_mirex2015 data, i.e. take different datasets from it, build several
ML models, and test their accuracies on the individual datasets'''



from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import *
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import scipy
#import pyaudio
#import struct
import numpy as np
import pandas as pd
import seaborn as sns

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

from scipy import signal
from scipy.io import wavfile

'''We test on the music_mirex2015 data, i.e. take different datasets from it, build several
ML models, and test their accuracies on the individual datasets'''



from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import scipy
#import pyaudio
#import struct
import numpy as np
import pandas as pd
import seaborn as sns

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

from scipy import signal
from scipy.io import wavfile


import os


'''Below, we read all the files seperately, one by one'''

#folder=r'C:\Users\PC\Documents\Important documents\Batvoice projects\muspeak_mirex2015_detection_examples'
folder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'
file1='/ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.csv'
full_filename1=folder + file1
data1=pd.read_csv(full_filename1, sep=',', header=None)

file2='/ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.csv'
full_filename2=folder + file2
data2=pd.read_csv(full_filename2, sep=',', header=None)

file3='/ConscinciasParalelasN11-OEspelhoEOReflexoFantasiasEPerplexidadesParte413-12-1994.csv'
full_filename3=folder + file3
data3=pd.read_csv(full_filename3, sep=',', header=None)

file4='/eatmycountry1609.csv'
full_filename4=folder + file4
data4=pd.read_csv(full_filename4, sep=',', header=None)

file5='/theconcert2_v2.csv'
full_filename5=folder + file5
data5=pd.read_csv(full_filename5, sep=',', header=None)

file6='/theconcert16.csv'
full_filename6=folder + file6
data6=pd.read_csv(full_filename6, sep=',', header=None)

file7='/UTMA-26_v2.csv'
full_filename7=folder + file7
data7=pd.read_csv(full_filename7, sep=',', header=None)

'''Below, we concatenate all these datasets to form a single pd dataframe'''
data_combined=pd.concat((data1, data2, data3, data4, data5, data6, data7)) 

'''Constructing the data matrices from the combined dataframe'''
X=data_combined.values[:, 0:2].astype(float)
Y=data_combined.values[:, 2]

'''We just work with the data1, date2 etc. individually here'''
X=data_combined.values[:, 0:2].astype(float)
Y=data_combined.values[:, 2]


'''train test split, and rescaling of the original data, but no PCA here (done later)'''
test_size = 0.20
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, shuffle=True)
X_train=preprocessing.scale(X_train)
X_test=preprocessing.scale(X_test)


'''chck for class imbalance in the total and test dataset'''
print('ratio of music to speech in the combined data is: ' \
      + str(Y[Y=='m'].shape[0]/Y[Y=='s'].shape[0]))


print('ratio of music to speech in the train part of combined data is: ' \
      + str(Y_train[Y_train=='m'].shape[0]/Y_train[Y_train=='s'].shape[0]))

print('ratio of music to speech in the test part of combined data is: ' \
      + str(Y_test[Y_test=='m'].shape[0]/Y_test[Y_test=='s'].shape[0]))
'''model building below'''
'''model building'''
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

'''Below, we test the above models with default values to get idea of accuracies'''
results = []
names = []
scoring= 'accuracy'      
for name, model in models:
    #below, we do k-fold cross-validation
	 kfold = model_selection.KFold(n_splits=10, shuffle=True) 
	 cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	 results.append(cv_results)
	 names.append(name)
	 msg = "%s: %f (%f)" % ('For combined data, mean and std of the cv accuracies for ' + name,  cv_results.mean(),  cv_results.std() );print(msg)

print("\n End of code \n")
