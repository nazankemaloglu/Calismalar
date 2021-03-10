import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score
window=5



#train and test data set loading

d = pd.read_excel('agri.xlsx')
d1=d.to_numpy()

data=d1.flatten()
data=pd.DataFrame(data,columns=['indis'])
data=data.dropna()

indexNames = data[ (data['indis'] == -9999) ].index
data.drop(indexNames , inplace=True)
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

from sklearn.svm import SVR  
svr = SVR(kernel='rbf', gamma=0.1) 
model = svr.fit(X_train, y_train)


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
y_test_predict=y_test_predict.reshape(-1,1)
true_prediction = sc.inverse_transform(y_test_predict)
draw=pd.concat([pd.DataFrame(testX),pd.DataFrame(true_prediction)],axis=1);
draw.iloc[:,0].plot(figsize=(12,6))
draw.iloc[:,1].plot(figsize=(12,6))
plt.legend(('Test', 'Test Tahmin'),loc='upper right',fontsize='15')
plt.title("Van Ortalama Sıcaklık",fontsize='30') #添加标题


from math import sqrt
r2=r2_score(df_test[:,:], y_test_predict)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
r2=r2_score(df_test[:,:], y_test_predict)

print("Accuacy ",r2)
print("RMSE ", sqrt(mean_squared_error(y_test_predict, y_test)))
print("MAE ",mean_absolute_error(y_test_predict, y_test))
print("MSE ", mean_squared_error(y_test_predict, y_test) )

#color =['Red', 'Blue']
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()

y = testX
fig, ax = plt.subplots()
#ax.scatter(y, true_prediction, c=color)
ax.scatter(y, true_prediction)

ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Test')
ax.set_ylabel('Test Tahmin')
plt.show()