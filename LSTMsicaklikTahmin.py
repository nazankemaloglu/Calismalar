import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import r2_score

data = pd.read_csv('agrimin.csv')


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

model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))


model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))


model.add(LSTM(units=50))
model.add(Dropout(0.2))


model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

results=model.fit(X_train, y_train, epochs=30, batch_size=4)

y_train_predict=model.predict(X_train)[:,0]
y_train=y_train


draw=pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_train_predict)],axis=1)
draw.iloc[100:400,0].plot(figsize=(12,6))
draw.iloc[100:400,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Train Data",fontsize='30') #添加标题
'''
history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=4)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
# summarize history for loss
plt.plot(results.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['Eğitim', 'Değerlendirme '], loc='upper left')
plt.show()

#test
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

predicted_stock =model.predict(X_test)
y_test_predict=model.predict(X_test)[:,0]
y_test=df_test[:,:]
true_prediction = sc.inverse_transform(predicted_stock)
true_prediction[:,0]

draw=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_test_predict)],axis=1);
draw.iloc[:,0].plot(figsize=(12,6))
draw.iloc[:,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Test Data",fontsize='30') #添加标题

r2=r2_score(df_test[:,:], predicted_stock)
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from math import sqrt

mse=mean_squared_error(df_test[:,:],predicted_stock)
rmse = sqrt(mse)
mae=mean_absolute_error(df_test[:,:],predicted_stock)
#visualize
plt.plot(testX[:,:], color = 'red', label='Test ')
plt.plot(true_prediction, color = 'blue', label='Test Tahmin')
plt.title('Van İli Ortalama Sıcaklık Tahmini')
plt.xlabel('Zaman (Gün)')
plt.ylabel('Ortalama Sıcaklık')
plt.legend()
plt.show()



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