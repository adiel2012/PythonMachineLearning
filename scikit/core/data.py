import numpy as np
import os
from scipy.io import arff
from io import StringIO


def get_full_data():
    url = os.path.dirname(__file__) + '\\arffs\\iris.arff'
    content =  open( url, 'r').read()
    f = StringIO(content)
    data, meta = arff.loadarff(f)
    X = np.zeros([len(data), len(data[0])-1])
    
    classes = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = data[i][j]
        
        class_value = data[i][ len(data[0])-1 ]
        if((class_value  in classes) == False):
            classes.append(class_value)

    Y = np.zeros([X.shape[0], len(classes)])
    for i in range(X.shape[0]):
        class_value = data[i][ len(data[0])-1 ]
        Y[i][classes.index(class_value)] = 1

    return X, Y


#X, Y = get_full_data()
#print(X)
#print(Y)