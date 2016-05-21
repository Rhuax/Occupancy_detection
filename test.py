from Dataset_reader2 import load_testset
from keras.models import model_from_json
import numpy as np
"""
Caricamento da file di una rete neurale già addestrata.
"""
network=model_from_json(open('Saved_Networks/1x30x0.001xrelu_epoch=100/structure.txt').read())
network.load_weights('Saved_Networks/1x30x0.001xrelu_epoch=100/weights.h5')
network.compile(optimizer='SGD',loss='mse')

#Test set loading
test_set=load_testset()
X=test_set[:,0:6]
y=test_set[:,6]

#Predizione
y_prediction=np.around(network.predict_on_batch(X)).flatten()
true_positives = np.sum(y_prediction*y)
precision= true_positives /np.sum(y_prediction)
recall = true_positives /np.sum(y)

true_negatives=0
for i in range(np.shape(y_prediction)[0]):
    if y_prediction[i]==0 and y[i]==0:
        true_negatives+=1

accuracy= (true_positives+true_negatives)/ np.shape(y)[0]
specificity= true_negatives/(np.shape(y)[0]-np.sum(y))
print('accuracy:'+str(accuracy)+' precision:'+str(precision)+' recall:'+str(recall)+' specificity:'+str(specificity))