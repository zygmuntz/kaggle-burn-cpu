# convert train file to numbers only

import sys
import pandas as pd

try:
	input_file = sys.argv[1]
except IndexError:
	input_file = 'train.csv'
	
try:
	output_file = sys.argv[2]
except IndexError:
	output_file = 'train_num.csv'	

data = pd.read_csv( input_file )

m_dummies = pd.get_dummies( data.m_id )
newdata = data.drop( ['m_id','sample_time'], axis = 1 )

alldata = newdata.join( m_dummies )

alldata.to_csv( output_file, index = None )
