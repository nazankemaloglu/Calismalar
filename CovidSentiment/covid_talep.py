import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from keras.models import Sequential
from keras.layers import Dense,LSTM,Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from numpy import argmax


data1=pd.read_csv('iki_data.csv')
data1=data1.drop(columns=['Unnamed: 0'])

train=[]
train1=[]

talep=data1[data1['2']=='Talep' ]
sikayet=data1[data1['2']=='Sikayet' ]
ovgu=data1[data1['2']=='Ovgu' ]
soru=data1[data1['2']=='Soru' ]
diger=data1[data1['2']=='Diger' ]

frame1 = [talep,sikayet,ovgu,soru,diger]
df = pd.concat(frame1)
cizim=df[df.columns[2]].value_counts()
train=[]
y=[]

for i in df['2']:
    if i=='Talep':
        y.append(0)
    elif i=='Sikayet':
        y.append(1)
    elif i=='Ovgu':
        y.append(2)
    elif i=='Soru':
        y.append(3)
    elif i=='Diger':
        y.append(4)
        
for word in df['0']:
    word=word.upper() 
    word=re.sub(r'@[A-Za-z0-9]+','',word)
    word=re.sub(r'#[A-Za-z0-9ğüşöçiıĞÜŞÖÇİI]+','',word)
    word=re.sub("(\w+:\/\/\S+)", "", word)
    word=re.sub("[^[a-zA-Z0-9ğüşöçiıĞÜŞÖÇİI]",' ',word)
    word= ''.join([i for i in word if not i.isdigit()])
    word=' '.join([j for j in word.split() if len(j)>2])
    word=word.lower()
    train.append(word)
    
t_r=pd.DataFrame(train,columns=['g'])    
y_tr=pd.DataFrame(y,columns=['f'])  
X_t  = result = pd.concat([t_r, y_tr], axis=1, join='inner')    
  
X_t['g'].replace('', np.nan, inplace=True)
X_t.dropna(subset=['g'], inplace=True)  

X=X_t['g']
y=X_t['f']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#ANN MODEL

labels = ('Talep','Sikayet','Övgü','Soru','Diger')
colors = ['yellow','lightblue','green','pink','purple']

import pandas as pd
import numpy as np

length = []
ind=[]
i=0
for x in X_train:
    length.append(len(x.split()))
    if len(x.split())<=1:
        ind.append(i)
    i=i+1

somelist = [i for j, i in enumerate(y_train) if j not in ind]
somelis = [i for j, i in enumerate(X_train) if j not in ind]


y_train=somelist
X_train=somelis

cat_ytrain = to_categorical(y_train, num_classes=5)
cat_ytest = to_categorical(y_test, num_classes=5)        
kelime_sayisi=max(length)
 
#top_words = len(length)
top_words=20000
num =kelime_sayisi
tk = Tokenizer(num_words=top_words, split=" ")
tk.fit_on_texts(X_train)
X_train_seq = tk.texts_to_sequences(X_train)
X_test_seq = tk.texts_to_sequences(X_test)

X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=num)
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=num)

model1=Sequential()
model1.add(Embedding(top_words, 256, input_length=num))
model1.add(Flatten())
model1.add(Dense(5,activation='softmax'))
model1.compile(loss='categorical_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])
model1.fit(X_train_seq_trunc,cat_ytrain, epochs=10, batch_size=128)
scores1=model1.evaluate(X_test_seq_trunc,cat_ytest)
score1 = model1.predict(X_test_seq_trunc)
print("-----------Result of Classical Artificial Neural Network--------------")
print("Accuracy: %.2f%%" % (scores1[1]*100))

cnn_y=argmax(score1,-1)
print(classification_report(y_test, cnn_y))
print(accuracy_score(y_test, cnn_y))
cnf_matrix = confusion_matrix(y_test, cnn_y)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cnf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# LSTM CLASSIFIER

model=Sequential()
model.add(Embedding(top_words, 256, input_length=num))
model.add(LSTM(16))
model.add(Dense(5, activation='softplus'))
model.compile(loss='categorical_crossentropy', 
             optimizer='adamax', 
             metrics=['accuracy'])
model.fit(X_train_seq_trunc,cat_ytrain, epochs=10,batch_size=128)
score1 = model.predict(X_test_seq_trunc)
lstm_y=argmax(score1,-1)
cm=confusion_matrix(y_test,lstm_y)
print("-----------Result of LSTM--------------")
print("LSTM Test= ",accuracy_score(y_test, lstm_y))
print(classification_report(y_test, lstm_y))

ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()