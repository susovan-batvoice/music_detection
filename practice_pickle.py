import pickle
example_dict={1:2, 2:'f', 3:'car', 4:[0,8,7]}
pickle_out=open('dict.pickle', 'wb')
pickle.dump(example_dict,pickle_out)
pickle_out.close()
pickle_in=open('dict.pickle', 'rb')
example_dict=pickle.load(pickle_in)
print(example_dict)