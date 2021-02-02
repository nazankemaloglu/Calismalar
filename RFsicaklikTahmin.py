import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score

#train and test data set loading

data = pd.read_csv('vanmin2.csv')

cut=100
k=len(data)
trainX = data.iloc[:k-100,:].values
testX = data.iloc[k-100:,:].values

sc = MinMaxScaler()

df_train = sc.fit_transform(trainX)
df_test = sc.fit_transform(testX)


X_train =[]
y_train = []

for i in range(60, k-100):
    ara = df_train[i-60:i, :]
    ara=ara.reshape(1, ara.shape[0]*ara.shape[1])
    X_train.append(ara)
    y_train.append(df_train[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[2], 1))

window=60 
X_train=X_train.reshape(len(X_train),window)
y_train=y_train.reshape(len(X_train))

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
model = rf.fit(X_train, y_train)


dataset_total = data
inputs = dataset_total[len(dataset_total) - len(df_test) -60:].values

inputs = sc.transform(inputs)

X_test =[]

for i in range(60, len(inputs)):
    ara = inputs[i-60:i, :]
    ara=ara.reshape(1, ara.shape[0]*ara.shape[1])
    X_test.append(ara)
    
    
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[2], 1))

X_test=X_test.reshape(cut,window)
y_test=df_test.reshape(cut)
y_test_predict=model.predict(X_test)
draw=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_test_predict)],axis=1);
draw.iloc[:,0].plot(figsize=(12,6))
draw.iloc[:,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Test Data",fontsize='30') #添加标题


from math import sqrt
r2=r2_score(df_test[:,:], y_test_predict)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
r2=r2_score(df_test[:,:], y_test_predict)
print('测试集上的MAE/MSE/MAPE')
print(r2)
print(sqrt(mean_squared_error(y_test_predict, y_test)))
print(mean_absolute_error(y_test_predict, y_test))
print(mean_squared_error(y_test_predict, y_test) )
print(mape(y_test_predict, y_test) )