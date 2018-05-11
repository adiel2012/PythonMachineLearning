import os
import sys

import numpy as np
#import matplotlib.pyplot as plt
import math
from sklearn.model_selection import cross_val_score



scriptpath = "../core/utils.py"

# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))


#params_collection = {
#    'a': [3,54,5], 
#    'b': [7,6]
#}
#
def testFunc(**kwargs):
    print(kwargs)    # prints the dictionary of keyword arguments
    mlp =  alg( **kwargs )
    scores = cross_val_score(mlp, X, Y, cv=folds)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../core")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../configs")


import utils
#utils.cross(params_collection, testFunc)

import data
import output

X,Y = data.get_full_data()

import mlp


configurations = mlp.getconfig()
folds = configurations['folds']
configs = configurations['configurations']

for config in configs:
    alg = config['algorithm_name']
    utils.cross(config['parameters'], testFunc)

