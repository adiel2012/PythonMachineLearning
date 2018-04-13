import os
import sys

import numpy as np
#import matplotlib.pyplot as plt
import math
#from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor


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
    mlp = MLPRegressor( kwargs )
    mlp.fit(X, Y)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../core")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../configs")

import utils
#utils.cross(params_collection, testFunc)

import data

X,Y = data.get_full_data()

import mlp
utils.cross(mlp.getconfig(), testFunc)

