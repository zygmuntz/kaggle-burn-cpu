import pandas as pd
from sklearn.preprocessing import StandardScaler as Scaler

input_file = 'data/train_num.csv'

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
