'''

Read the CSV files and folders corresponding to music and make dfs and matrices
'''
import numpy as np
import pandas as pd
import os


def df_et_matrix_one_file(folder,file):

    '''


    :param folder: str
    :param file: str, CSV format
    :return: dataframe, array, array
    '''

    #full_filename = folder + file
    full_filename = os.path.join(folder, file)
    df=pd.read_csv(full_filename, sep=',', header=None)
    X = df.values[:, 0:2].astype(float)
    Y = df.values[:, 2]
    return df, X, Y

def df_et_matrix_multiple_files(folder,files):
    """

    :param args_folders: [str]
    :param args_files: [str]
    :return:
    """

    df_list=[]
    X_list=[]
    Y_list=[]

    for file in files:
            #full_filename = folder + file
            full_filename=os.path.join(folder,file)
            df = pd.read_csv(full_filename, sep=',', header=None)
            X = df.values[:, 0:2].astype(float)
            Y = df.values[:, 2]
            df_list.append(df)
            X_list.append(X)
            Y_list.append(Y)
    return df_list, X_list, Y_list



'''
#TEST the above fn

#folder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'
#file1='ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.csv'
#print(df_et_matrix_one_file(folder,file1))


folder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'
file2='ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.csv'
print(df_et_matrix_one_file(folder,file2))

'''

'''

#Test the above fn

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

folder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'
files= [file1, file2, file3]

for file in files:
    print(  '\n the df and arrays for' + str(file)  + 'are \n'  + str(df_et_matrix_multiple_files(folder,files)) + '\n')

'''