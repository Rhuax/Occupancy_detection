import numpy as np


training_set_path='datasets/training_set.csv'
test_set_path='datasets/data_test.csv'

"""
Caricamento del training set e rimescolata
"""
def load_training_set():
    print('Loading training set..')
    array=np.genfromtxt(training_set_path,delimiter=',')
    np.random.shuffle(array)
    print('Training set loaded, shape :'+str(np.shape(array)))
    return array

"""
Caricamento del test set
"""
def load_testset():
    print('Loading test set..')
    array2=np.genfromtxt(test_set_path,delimiter=',')
    print('Test set loaded,shape : '+str(np.shape(array2)))
    return array2    