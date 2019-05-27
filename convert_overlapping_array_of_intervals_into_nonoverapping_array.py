'''This function takes as
INPUT: an N x 2 np array of floats, where each row denotes the start 
time and duration of a music or speech, and 
OUTPUT: a different N x 2 np array where
the music and strings don't overapand 0-th element of thos row indcates the new start time
and 1st element the nex end time. The new start and end times are obtained by replacing
the previous start and end times of the music/speech if their end times were bigger than
the start times of the next music/speech, denoting an overap. Note that M can be equal to N'''

import numpy as np
np.set_printoptions(precision=10)

#array=data2.values[:, 0:2].astype(float)

non_overlapping_rows_list=[]
def make_non_overlapping_rows_from_overlapping_rows(array):
    if array.shape[1]!=2:
        print('Your array has wrong dimension, second dimenssion must be 2')
    else:
        for i in range(0, len(array)-1):
            if array[i,0]+array[i,1] > array[i+1,0] and array[i,0]!=array[i+1,0]: #if successive music or speech overlaps 
               #non_overlapping_array=np.delete(array,i,0) #delete i-th row
               non_overlapping_rows_list.append(np.array( [ array[i,0],array[i+1,0] ] ))
    
            else:
                non_overlapping_rows_list.append(np.array( [ array[i,0],array[i,0] + array[i,1] ] ))
    non_overlapping_rows_list.append(np.array( [ array[len(array)-1,0],array[len(array)-1,0] + array[len(array)-1,1] ] )) #finally, add the last row           
    non_overlapping_array=np.array(non_overlapping_rows_list) #converting list into np array
    return non_overlapping_array
#    for k in range(0, len(non_overlapping_array)-1):
#        if non_overlapping_array[k,0]==non_overlapping_array[k,1]:
#           non_overlapping_array=np.delete(non_overlapping_array,k,0)              
#    return non_overlapping_array        
#        
   
#X=make_non_overlapping_rows_from_overlapping_rows(array); 
#print('After taking care of overap, new array is: \n' + str(X))
    
