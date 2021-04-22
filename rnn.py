import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import r2_score

d = pd.read_excel('C:/Users/BT-NAZAN/Downloads/bitlis.xlsx')
d1=d.to_numpy()

data1=d1.flatten()
data1=pd.DataFrame(data1,columns=['indis'])
data1=data1.dropna()

indexNames = data1[ (data1['indis'] == -9999) ].index
data1.drop(indexNames , inplace=True)
#data=data1.replace(-9999, 0)

df_new=data1


x = len(df_new)-5

train=df_new.iloc[:x]
test = df_new.iloc[x:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train) #find max value

MinMaxScaler(copy=True, feature_range=(0, 1))

scaled_train = scaler.transform(train)#and divide every point by max value
scaled_test = scaler.transform(test)
print(scaled_train[-5:])
from keras.preprocessing.sequence import TimeseriesGenerator

n_input = 5  ## number of steps
n_features = 1 ## number of features you want to predict (for univariate time series n_features=1)
generator = TimeseriesGenerator(scaled_train,scaled_train,length = n_input,batch_size=1)

x,y = generator[50]
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation

model = Sequential()
model.add(LSTM(75,activation="relu",input_shape=(n_input,n_features)))
#model.add(Dropout(0.2))
model.add(Dense(75, activation='relu'))
model.add(Dense(units=1))
#model.add(Activation('softmax'))
#model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")

validation_set = np.append(scaled_train[55],scaled_test)
validation_set=validation_set.reshape(6,1)

n_input = 5
n_features = 1
validation_gen = TimeseriesGenerator(validation_set,validation_set,length=5,batch_size=1)

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)


model.fit_generator(generator,validation_data=validation_gen,epochs=100,steps_per_epoch=10)
pd.DataFrame(model.history.history).plot(title="loss vs epochs curve")

myloss = model.history.history["val_loss"]
plt.title("validation loss vs epochs")
plt.plot(range(len(myloss)),myloss)

test_prediction = []

##last n points from training set
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape(1,n_input,n_features)

for i in range(len(test)+7):
    current_pred = model.predict(current_batch)[0]
    test_prediction.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
true_prediction = scaler.inverse_transform(test_prediction)
true_prediction[:,0]

time_series_array = test.index
for k in range(0,7):
    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))
time_series_array
df_forecast = pd.DataFrame(columns=["dead","dead_predicted"],index=time_series_array)
df_forecast.loc[:,"dead_predicted"] = true_prediction[:,0]
df_forecast.loc[:,"dead"] = test["dead"]
df_forecast.plot(title="US Predictions for next 7 days")
MAPE = np.mean(np.abs(np.array(df_forecast["dead"][:5]) - np.array(df_forecast["dead_predicted"][:5]))/np.array(df_forecast["dead"][:5]))
print("MAPE is " + str(MAPE*100) + " %")
sum_errs = np.sum((np.array(df_forecast["dead"][:5]) - np.array(df_forecast["dead_predicted"][:5]))**2)
stdev = np.sqrt(1/(5-2) * sum_errs)
interval = 1.96 * stdev

#df_forecast["confirm_min"] = df_forecast["confirmed_predicted"] - interval
#df_forecast["confirm_max"] = df_forecast["confirmed_predicted"] + interval

df_forecast["Model Accuracy"] = round((1-MAPE),2)

from datetime import datetime
df_forecast["Country"] = country
df_forecast["Execution date"] = str(datetime.now()).split()[0]
df_forecast.iloc[:,:4].plot()

fig= plt.figure(figsize=(10,5))
plt.title("{} - Results".format(country))
plt.plot(df_forecast.index,df_forecast["dead"],label="dead")
plt.plot(df_forecast.index,df_forecast["dead"],label="dead")
#ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)
#plt.fill_between(df_forecast.index,df_forecast["confirm_min"],df_forecast["confirm_max"],color="indigo",alpha=0.09,label="Confidence Interval")
plt.legend()
plt.show()