import numpy as np
from Dataset_reader2 import load_training_set
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

fold_size = 2000
K_fold=4
Total_accuracy=0
Total_precision=0
Total_specificity=0
Total_recall= 0
Best_accuracy = 0
Best_network = None

"""
Slice del training set dedicata al validation
"""
def get_validation_split(X,k):
    KValidX = X[k * fold_size:(k + 1) * fold_size,0:6]
    KYValidX = X[k * fold_size:(k + 1) * fold_size,6]
    return KValidX,KYValidX


"""
Slice del training set dedicata all'addestramento del k-esimo cross validation
"""
def get_training_split(X,k):
    if k == 0:
        KTrainX = X[(k + 1) * fold_size:,0:6]
        KYX = X[(k + 1) * fold_size:, 6]
    elif k == K_fold - 1:
        KTrainX = X[0:k * fold_size,0:6]
        KYX = X[0:k * fold_size, 6]
    else:
        KTrainX = np.concatenate((X[0:k * fold_size,0:6], X[(k + 1) * fold_size:,0:6]),axis=0)
        KYX = np.concatenate((X[0:k * fold_size, 6], X[(k + 1) * fold_size:, 6]), axis=0)
    return KTrainX,KYX

"""
Definizione della struttura della rete neurale, inizializzazione e compilazione
"""
def build_neural_network():
    network= Sequential()
    network.add(Dense(30,init='uniform',activation='relu',input_dim=6))
    """network.add(Dense(20,init='uniform',activation='tanh'))"""
    network.add(Dense(1,init='uniform',activation='sigmoid'))
    sgd=SGD(lr=0.001)
    network.compile(sgd,loss='mse')
    return network

"""
Funzione dedicata alla valutazione di ogni rete neurale su un validation set.
Metriche:
    Precision,Recall,Accuracy,Sensitivity
Ogni metrica viene salvata e viene tenuto traccia della rete neurale più
performante.
"""
def evaluate_on_validation_set(network,X,y):

    y_prediction=np.around(network.predict_on_batch(X)).flatten()
    true_positives = np.sum(y_prediction*y)

    if true_positives==0:
        precision=0
        recall=0
    else:
        precision= true_positives /np.sum(y_prediction)
        recall = true_positives /np.sum(y)


    true_negatives=0
    for i in range(np.shape(y_prediction)[0]):
        if y_prediction[i]==0 and y[i]==0:
            true_negatives+=1

    accuracy= (true_positives+true_negatives)/ np.shape(y)[0]
    specificity= true_negatives/(np.shape(y)[0]-np.sum(y))
    print('accuracy:'+str(accuracy)+' precision:'+str(precision)+' recall:'+str(recall)+
    ' specificity:'+str(specificity))


    global Total_accuracy,Total_recall,Total_precision,Best_accuracy,Best_network,Total_specificity
    Total_accuracy+=accuracy
    Total_precision+=precision
    Total_recall+=recall
    Total_specificity+=specificity
    if accuracy > Best_accuracy:
        Best_accuracy=accuracy
        Best_network= network


training_set = load_training_set()

"""
K-fold cross validation
"""
for k in range(K_fold):
    print('K fold number :'+str(k))
    X,y= get_training_split(training_set,k)
    net= build_neural_network()
    #Training
    net.fit(X,y,batch_size=100,nb_epoch=150,verbose=1,shuffle=True)
    ValidX,ValidY= get_validation_split(training_set,k)
    print('Evaluating..')
    evaluate_on_validation_set(net,ValidX,ValidY)



"""
Salvataggio della rete neurale con miglior performance
"""
f =open('Saved_Networks/1x30x0.001xrelu_epoch=150/logs.txt','w')
f.write('Accuracy : '+str(Total_accuracy/K_fold)+'\n Precision: '+str(Total_precision/K_fold)
        +'\n Recall :'+str(Total_recall/K_fold)+'\n Specificity : '+str(Total_specificity/K_fold))
f.close()

json= Best_network.to_json()
f= open('Saved_Networks/1x30x0.001xrelu_epoch=150/structure.txt','w')
f.write(json)
f.close()

Best_network.save_weights('Saved_Networks/1x30x0.001xrelu_epoch=150/weights.h5')




