"load a few best settings, train the models, average predictions"

import numpy as np
import pandas as pd

from elm import *
from random_layer import *

from sklearn import pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as MSE

from pipeline_driver import RMSE

from load_data import x_train, y_train, x_test, y_test

def get_predictions( params ):
	
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

	return p

input_file = 'hyperopt_logs/hyperopt_log_pipeline.csv'

#

data = pd.read_csv( input_file )

best = data[data.rmse < 5]

print "training {} models...".format( len( best ))

all_p = 'dummy'

for params in best.itertuples():
	
	# 0 is index
	rmse = params[1]
	params = params[2:]
	#print rmse, params
	
	p = get_predictions( params )
	new_rmse = RMSE( y_test, p )
	
	print "\n{}\t{}\n".format( rmse, new_rmse )
	
	p = p.reshape( -1, 1 )
	
	try:
		all_p = np.hstack(( all_p, p ))
	except ValueError:
		all_p = p
		
		
assert( all_p.shape[1] == len( best ))		

p_bag = np.mean( all_p, axis = 1 )
bag_rmse = RMSE( y_test, p_bag )

print "bagged RMSE:", bag_rmse

"""
4.80702905174	4.86961396002

4.69688727788	4.71511820023

4.65203836026	4.76711428554

4.90157856325	5.04205983981

4.88192059806	4.90897360006

4.6670679981	4.7602996695

4.79311069383	4.81340706305

4.6723412506	4.65207347166

bagged RMSE: 4.49256467128
"""


