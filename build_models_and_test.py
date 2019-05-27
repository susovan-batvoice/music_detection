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
import pickle
import operator

def train_test_split(X,Y,test_size=0.20):
    '''train test split, and rescaling of the original data, but no PCA here (done later)'''
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, shuffle=True)
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    #check for class imbalance below
    print('\n in the original data, the ratio of music to speech is ' + str(Y[Y=='m'].shape[0]/Y[Y=='s'].shape[0]))
    print('\n in the training data, the ratio of music to speech is ' + str(Y_train[Y_train == 'm'].shape[0] / Y_train[Y_train == 's'].shape[0]))
    print('\n in the test data, the ratio of music to speech is ' + str(Y_test[Y_test == 'm'].shape[0] / Y_test[Y_test == 's'].shape[0]))
    return X_train, Y_train, X_test, Y_test



'''
#Test the above fn:
pickle_in=open('dict.pickle', 'rb')
data_dict=pickle.load(pickle_in)
X,Y=data_dict[1], data_dict[2]
X_train, Y_train, X_test, Y_test=train_test_split(X,Y,test_size=0.20)
print("\n After train test split, training MFCC are:\n " + str(X_train))
print("\n After train test split, training annotations are are:\n " + str(Y_train))
print("\n After train test split, test MFCC are:\n " + str(X_test))
print("\n After train test split, test annotations are:\n " + str(Y_test))

'''


def perform_cross_validation_for_one_model(X,Y):
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, shuffle=True)
    #X_train = preprocessing.scale(X_train)
    #X_test = preprocessing.scale(X_test)
    results = []
    names = []
    means_cv_results=[]
    scoring = 'accuracy'
    for name, model in models:
        # below, we do k-fold cross-validation
        kfold = model_selection.KFold(n_splits=10, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % ('mean and std of the cv accuracies for ' + name, cv_results.mean(), cv_results.std())
        means_cv_results.append(cv_results.mean())
        print(msg)
        #index, value = max(enumerate(means_cv_results),
    max_value = max(means_cv_results)
    max_index = means_cv_results.index(max_value)
    print('\n cv accuracy is maximized at \n' + str(models[max_index][1]))

    return cv_results, names, models[max_index][1]

'''
#TEST the above fn:
pickle_in=open('dict.pickle', 'rb')
data_dict=pickle.load(pickle_in)
X,Y=data_dict[1], data_dict[2]
perform_cross_validation_for_one_model(X,Y)

'''


def prediction_results(X,Y,model):
    X_train, Y_train, X_test, Y_test=train_test_split(X,Y,test_size=0.20)
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    model.fit(X_train, Y_train)
    pred_train= model.predict(X_train)
    print('\n training accuracy for ' + str(model) + 'before PCA and hyp. opt. is '  +  str(accuracy_score(Y_train, pred_train)) )
    predictions = model.predict(X_test)
    print('\n test accuracy for ' + str(model) + 'before PCA and hyp. opt. is '  +  str (accuracy_score(Y_test, predictions))  )
    print('\n confusion matrix for ' + str(model) + 'before PCA and hyp. opt. is \n' + str(confusion_matrix(Y_test, predictions)) )
    print('\n detailed classification results for test data, before PCA and hyp. opt. are \n' + str(classification_report(Y_test, predictions)) )




def pred_after_PCA(X,Y,model):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, shuffle=True)
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    print('PERFORMING PCA \n')
    pca = PCA(n_components=1)
    #pca = PCA(.95)
    pca.fit(X_train)
    X_train_PCA = pca.transform(X_train)
    X_test_PCA = pca.transform(X_test)
    model.fit(X_train_PCA, Y_train)
    pred_train_PCA = model.predict(X_train_PCA)
    pred_test_PCA = model.predict(X_test_PCA)
    print('\n training accuracy for' +  str(model)  +  'after PCA is '  +  str(accuracy_score(Y_train, pred_train_PCA)) )
    print('\n test accuracy for' + str(model) + 'after PCA is ' + str(accuracy_score(Y_test, pred_test_PCA)))
    predictions = model.predict(X_test_PCA)
    print('\n test accuracy for' + str(model) + 'after PCA  is '  +  str (accuracy_score(Y_test, predictions))  )
    print('\n confusion matrix for' + str(model) + 'after PCA is \n' + str(confusion_matrix(Y_test, predictions)) )
    print('\n detailed classification results for test data, after PCA are \n' + str(classification_report(Y_test, predictions)) )


'''
#Test the above fn:
pickle_in=open('dict.pickle', 'rb')
data_dict=pickle.load(pickle_in)
X,Y=data_dict[1], data_dict[2]
X_train, Y_train, X_test, Y_test=train_test_split(X,Y,test_size=0.20)
print("\n After train test split, training MFCC are:\n " + str(X_train))
print("\n After train test split, training annotations are are:\n " + str(Y_train))
print("\n After train test split, test MFCC are:\n " + str(X_test))
print("\n After train test split, test annotations are:\n " + str(Y_test))
model = perform_cross_validation_for_one_model(X, Y)[2]  # gives the name of the model with best CV accuracy
prediction_results()
#print('debuggging finishes here')

'''


