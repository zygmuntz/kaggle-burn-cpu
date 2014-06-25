"re-run best settings from the log to check if they're stable"

import pandas as pd

from driver import run_test
from load_data import x_train, y_train, x_test, y_test

#input_file = 'hyperopt_logs/hyperopt_log_pipeline.csv'
input_file = 'hyperopt_logs/hyperopt_log.csv'

data = pd.read_csv( input_file )

best = data[data.rmse < 5]

print "running {} tests...".format( len( best ))

for params in best.itertuples():
	# 0 is index
	rmse = params[1]
	params = params[2:]
	#print rmse, params
	
	new_rmse = run_test( params )
	print "\n{}\t{}\n".format( rmse, new_rmse )
	

"""
pipeline

4.80702905174	4.7985210499

4.69688727788	4.66954784239

4.65203836026	4.65734020556

4.90157856325	5.18929028774

4.88192059806	4.92931446918

4.6670679981	4.69555385193

4.79311069383	4.87519996373

4.6723412506	4.68784431689
"""

"""
basic driver
4.99462852658	5.03561622165

4.76383638402	4.74549670353

4.76120992754	4.78784683856

4.72012502539	4.75416775177

4.85692787229	4.95283375459
"""

