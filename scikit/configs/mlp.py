from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC

out = { 
    'folds' : 5,
    'configurations' :[   
        {'parameters' : {
            'hidden_layer_sizes' : [(2,), (3,), (4,), (5,), (10,), (15,), (20,)  ] ,
            'max_iter' : [100, 200, 300, 500, 1000]
                        } , 
            'algorithm_name' : MLPRegressor
        },
        {'parameters' : {
            'C' : [0.1, 0.2, 0.5, 0.8, 1, 1.5 ] ,
            'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
                        } , 
            'algorithm_name' : SVC
        }

    ]
    
}

def getconfig():
    return out