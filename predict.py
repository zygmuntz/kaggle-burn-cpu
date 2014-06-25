"select best params from a hyperopt log, train, predict and save solution"

import numpy as np
import pandas as pd

from elm import *
from random_layer import *

from sklearn import pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler as Scaler

#

log_file = 'hyperopt_logs/hyperopt_log_pipeline.csv'
train_file = 'data/train_num.csv'
test_file = 'data/test_num.csv'
predictions_file = 'data/elm_predictions.csv'

# get params for the best model from the hyperopt_log

log_data = pd.read_csv( log_file )
best_i = log_data.rmse.argmin()
params = list( log_data.loc[best_i] )[1:]

print "params:"
print log_data.loc[best_i]

# load train

print "loading train..."

train = pd.read_csv( train_file )

y_train = train[ 'cpu_01_busy' ]
x_train = train.drop( 'cpu_01_busy', axis = 1 )

scaler = Scaler()
x_train = scaler.fit_transform( x_train )

# load test

print "loading test..."

test = pd.read_csv( test_file )

x_test = test.drop( 'Id', axis = 1 )
x_test = scaler.transform( x_test )

# train and predict

print "training..."

n_hidden, alpha, rbf_width, activation_func, ridge_alpha = params
n_hidden = int( n_hidden )

"""
print "n_hidden:", n_hidden
print "alpha:", alpha
print "rbf_width:", rbf_width
print "activation_func:", activation_func
print "ridge_alpha:", ridge_alpha
"""
	
rl = RandomLayer( n_hidden = n_hidden, alpha = alpha, 
	rbf_width = rbf_width, activation_func = activation_func )

ridge = Ridge( alpha = ridge_alpha )

elmr = pipeline.Pipeline( [( 'rl', rl ), ( 'ridge', ridge )] )	

elmr.fit( x_train, y_train )
p = elmr.predict( x_test )

# save predictions

predictions = pd.DataFrame( { 'Id': test.Id, 'Prediction': p } )

predictions.to_csv( predictions_file, index = None )

