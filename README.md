kaggle-burn-cpu
===============

Code for the [Burn CPU, burn](http://inclass.kaggle.com/c/model-t4) competition at Kaggle. Shows how to tune Extreme Learning Machines with [hyperopt](https://github.com/hyperopt/hyperopt). Uses [Python-ELM](https://github.com/dclambert/Python-ELM).

	`elm.py, random_layer.py` - Python-ELM files - see `Python-ELM-LICENSE`
	`data2num.py` - convert data to numbers only
	`..._driver.py` - different flavours of optimization scripts to run
	`hyperopt_logs` - results from each driver
	`predict.py` - select best params from a log, train and save predictions
	
	auxillary files:
	`bag_best.py` - select a few best models from a log, train and average their predictions (using the validation set)
	`rerun_best.py` - select a few best models from a log, re-run them to check the scores
	`load_data.py` - a module for loading data used by `bag_best.py` and `rerun_best.py`
	
	
License: BSD
