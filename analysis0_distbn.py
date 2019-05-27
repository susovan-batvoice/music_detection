'''
We perform some distributional analysis, e.g. compute mean, variance, kurtosis etc.



'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
def mean_cov_binary_cls(X, Y, str1, str2):

    '''
    :param X: np array , e.g. MFCC coeff, of size N * p
    :param Y: np array of strings, e;g. annotations str 1, str 2
    :param str1: annotation 1
    :param str2: annotation 2
    :return:
    '''
    mean_all_data = np.mean(X, axis = 0)
    cov_all_data = np.cov(X.T)
    #plt.imshow(cov_all_data, cmap='hot', interpolation='nearest')
    #plt.show()
    mean_all_data = np.mean(X, axis=0)
    cov_all_data = np.cov(X.T)
    # plt.imshow(cov_all_data, cmap='hot', interpolation='nearest')
    # plt.show()
    sns.heatmap(cov_all_data)
    pos_str1 = np.where(Y == str1)
    pos_str2 = np.where(Y == str2)
    X_str1 = X[pos_str1]
    X_str2 = X[pos_str2]
    mean_str1 = np.mean(X_str1, axis=0)
    mean_str2 = np.mean(X_str2, axis=0)
    cov_str1 = np.cov(X_str1.T)
    pos_str1 = np.where(Y == str1)
    pos_str2 = np.where(Y == str2)
    X_str1 = X[pos_str1]
    X_str2 = X[pos_str2]
    mean_str1 = np.mean(X_str1, axis=0)
    mean_str2 = np.mean(X_str2, axis=0)
    cov_str1 = np.cov(X_str1.T)
    cov_str2 = np.cov(X_str2.T)
    #plt.imshow(cov_str1, cmap='hot', interpolation='nearest')
    #plt.imshow(cov_str2, cmap='hot', interpolation='nearest')
    #plt.subplot(2, 1, 1)
    #plt.imshow(cov_str1, cmap='hot', interpolation='nearest')
    #plt.xlabel('Frame no')
    #plt.ylabel(f_names[0])
    #plt.subplot(2, 1, 2)
    #plt.imshow(cov_str1, cmap='hot', interpolation='nearest')
    #plt.plot(F[1, :])
    #plt.xlabel('Frame no')
    #plt.ylabel(f_names[1])
    #plt.show()
    sns.heatmap(cov_str1)
    sns.heatmap(cov_str2)
    print(  '\n covariance matrix for all data is \n' +  str(cov_all_data) )
    print( '\n covariance matrix for all' +  str(str1) + 'data is \n' +  str(cov_str1) )
    print('\n covariance matrix for all' + str(str2) + 'data is \n' + str(cov_str2))
    return mean_all_data, cov_all_data, mean_str1, cov_str1, mean_str2, cov_str2

def KS_test(arr1, arr2, p_value):
    '''
     2-sample KS test
    :param arr1: np array for the samples from the first distribution, flattened
    :param arr2: same , for the 2nd distbn
    :param p_value: given p-value
    :return:
    '''
    (statistic, pvalue) = stats.ks_2samp(arr1.flatten(), arr2.flatten())
    if pvalue < p_value:
        print('\n distributions are different \n')
    else:
        print( '\n distributions are similar \n')
    print('\n KS statistic is \n' + str(statistic))
    print('\n p value is \n' + str(pvalue))
    return statistic, pvalue


'''
#Test the above function:

try:
    foo_0 = pickle.load(open("model_dict0.pickle", "rb"))
except (OSError, IOError) as e:
    foo_0 = 1
    pickle.dump(foo, open("model_dict0.pickle", "wb"))
X0 = foo_0[2]
Y0 = foo_0[3]

mean_all_data, cov_all_data, mean_str1, cov_str1, mean_str2, cov_str2 = mean_cov_binary_cls(X0, Y0, 'm', 's')
sns.heatmap(cov_str1)

sns.heatmap(cov_str2)
'''