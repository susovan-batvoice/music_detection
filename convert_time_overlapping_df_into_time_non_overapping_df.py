'''Here, we convert the csv files or pd dataframe with overlapping music and speech intervals into another csv
csv files or pd dataframe where there's no more overlap'''

import numpy as np
import pandas as pd

#first way to modularise and import a function from the module
import make_df_et_matrix

# second way to modularise and import a function from the module
from make_df_et_matrix import df_et_matrix_one_file


np.set_printoptions(precision=1000)

#non_overlapping_rows_list=[]
def make_non_time_overlapping_df_from_time_overlapping_df(df):
    '''
    takes the df of the music containing starting and duration points of the music and speech,
    and produces a new df of starttime, endtime so that the music and speech don't overlap


    :param df: pd dataframe
    :return: pd dataframe
    '''




    start_time_list=[]
    end_time_list=[]
    #non_overlapping_rows_list=[]
    array = df.values[:, 0:2].astype(float)
    for i in range(0, len(array)-1):
       if array[i,0]+array[i,1] > array[i+1,0] and array[i,0]!=array[i+1,0]: #if successive music or speech overlaps 
        #non_overlapping_array=np.delete(array,i,0) #delete i-th row
         #non_overlapping_rows_list.append(np.array( [ array[i,0],array[i+1,0] ] ))
         start_time_list.append(array[i,0])
         end_time_list.append(array[i+1,0])
       else:
         #non_overlapping_rows_list.append(np.array( [ array[i,0],array[i,0] + array[i,1] ] ))
         start_time_list.append(array[i,0])
         end_time_list.append(array[i,0] + array[i,1])
    #non_overlapping_rows_list.append(np.array( [ array[len(array)-1,0],array[len(array)-1,0] + array[len(array)-1,1] ] )) #finally, add the last row
    start_time_list.append(array[len(array)-1,0])
    end_time_list.append(array[len(array)-1,0] + array[len(array)-1,1] )

    #non_overlapping_array=np.array(non_overlapping_rows_list) #converting list into np array
    for k in range(0, len(start_time_list)-1):
        if start_time_list[k]==end_time_list[k]:
           #non_overlapping_array=np.delete(non_overlapping_array,k,0)
           start_time_list.remove(k)
           end_time_list.remove(k)
    array2=df.values[:, 2] #keeping the original seq of 'm' and 's'
    #df=pd.DataFrame({'A': array, 'B': array2}, index=index)
    df2=pd.DataFrame({'A': start_time_list, 'B': end_time_list, 'C': list(array2) })
    return df2




#test the above fn
#folder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'
#file1='/ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.csv'

'''
folder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'
file2='ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.csv'


df=df_et_matrix_one_file(folder,file2)[0]
#folder='/home/susovan/Documents/music_detection/muspeak_mirex2015/muspeak-mirex2015-detection-examples'
X=make_non_time_overlapping_df_from_time_overlapping_df(df)
#print('After taking care of overlap, new dataframe is: \n\n' + str(X))
print(X)
print(type(X))

'''