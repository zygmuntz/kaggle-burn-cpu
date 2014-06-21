#!/usr/bin/env python

"tune ELM + linear regressor on top"

import pandas as pd
import numpy as np
import math
import csv
import hyperopt
import os
import sys

from time import time
from glob import glob
from math import log

from hyperopt import hp, fmin, tpe

from elm import *
from random_layer import *

from sklearn import pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler as Scaler

def RMSE( y, p ):
	return math.sqrt( MSE( y, p ))

###

input_file = 'data/train_num.csv'
try:
	output_file = sys.argv[1]
except IndexError:	
	output_file = 'hyperopt_log_pipeline.csv'

data = pd.read_csv( input_file )

train = data[0:93170]
test = data[93170:]

x_train = train[[ c for c in train.columns if c != 'cpu_01_busy' ]]
x_test = test[[ c for c in train.columns if c != 'cpu_01_busy' ]]

scaler = Scaler()
x_train = scaler.fit_transform( x_train )
x_test = scaler.transform( x_test )

y_train = train[ 'cpu_01_busy' ]
y_test = test[ 'cpu_01_busy' ]

#

max_evals = 50
run_counter = 0

def run_wrapper( params ):
	global run_counter
	global o_f
	
	run_counter += 1
	print "run", run_counter
	
	s = time()
	rmse = run_test( params )
	
	print
	print "RMSE:", rmse
	print "elapsed: {}s \n".format( int( round( time() - s )))

	writer.writerow( [ rmse ] + list( params ))
	o_f.flush()
	return rmse
	
def run_test( params ):
	
	n_hidden, alpha, rbf_width, activation_func, ridge_alpha = params
	n_hidden = int( n_hidden )
	
	print "n_hidden:", n_hidden
	print "alpha:", alpha
	print "rbf_width:", rbf_width
	print "activation_func:", activation_func
	print "ridge_alpha:", ridge_alpha
		
	rl = RandomLayer( n_hidden = n_hidden, alpha = alpha, 
		rbf_width = rbf_width, activation_func = activation_func )
	
	ridge = Ridge( alpha = ridge_alpha )
	
	elmr = pipeline.Pipeline( [( 'rl', rl ), ( 'ridge', ridge )] )	
	
	elmr.fit( x_train, y_train )
	p = elmr.predict( x_test )

	rmse = RMSE( y_test, p )
	return rmse
	


###

space = ( 
	hp.qloguniform( 'n_hidden', log( 10 ), log( 1000 ), 1 ),
	hp.uniform( 'alpha', 0, 1 ),
	hp.loguniform( 'rbf_width', log( 1e-5 ), log( 100 )),
	hp.choice( 'activation_func', [ 'tanh', 'sine', 'tribas', 'inv_tribas', 'sigmoid', 'hardlim', 'softlim', 'gaussian', 'multiquadric','inv_multiquadric' ] ),
	hp.loguniform( 'ridge_alpha', -15, 5 )
)

###

if __name__ == '__main__':

	headers = [ 'rmse', 'n_hidden', 'alpha', 'rbf_width', 'activation_func', 'ridge_alpha' ]
	o_f = open( output_file, 'wb' )
	writer = csv.writer( o_f )
	writer.writerow( headers )

	start_time = time()
	best = fmin( run_wrapper, space, algo = tpe.suggest, max_evals = max_evals )
	end_time = time()

	print "Seconds passed:", int( round( end_time - start_time ))
	#print "Best run:", optimizer.get_best_run()
	print best
	#print run_test( hyperopt.space_eval( space, best ))
