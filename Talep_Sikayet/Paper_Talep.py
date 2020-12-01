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

train=[]
train1=[]
dataframe=pd.read_excel('Talep.xlsx')
dataframe1=pd.read_excel('TalepVeri.xlsx')
Metin=dataframe[['Tweet','Durum']]
df1 = Metin.dropna(how = 'any')
Metin2=dataframe1[['Tweet','Durum']]
df2 = Metin2.dropna(how = 'any')
frame = [df1,df2]
resul = pd.concat(frame)
talep=resul[resul['Durum']=='Talep' ]
diger=resul[resul['Durum']=='Diger' ]
sikayet=resul[resul['Durum']=='Sikayet' ]


frame1 = [talep,sikayet,diger]
df = pd.concat(frame1)
cizim=df[df.columns[1]].value_counts()
train=[]
y=[]

for i in df['Durum']:
    if i=='Talep':
        y.append(0)
    elif i=='Diger':
        y.append(1)
    elif i=='Sikayet':
        y.append(2)

        
for word in df['Tweet']:
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

# svc CLASSİFİER TFIDF

from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from sklearn.ensemble import RandomForestClassifier
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

#TF-IDF
vect = TfidfVectorizer(ngram_range=(1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
norm = Normalizer().fit(X_train_vectorized)
x_train_tfidf_norm = norm.transform(X_train_vectorized)
x_test=vect.transform(X_test)
x_test_tfidf_norm = norm.transform(x_test)

model = svm.LinearSVC()
model.fit(x_train_tfidf_norm, y_train)
predictions = model.predict(x_test_tfidf_norm)
print("------------------TFIDFVektorizer SVC Sonuçları--------------")
print("TFIDFVektorizer SVC ngram skor",accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
cnf_matrix1 = confusion_matrix(y_test, predictions)
print("--------------------------------------------------------")
# SVC CLASSIFIER COUNTVECTORIZER

del vect,X_train_vectorized
vect = CountVectorizer(ngram_range=(1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
norm = Normalizer().fit(X_train_vectorized)
x_train_tfidf_norm = norm.transform(X_train_vectorized)
x_test=vect.transform(X_test)
x_test_tfidf_norm = norm.transform(x_test)

model = svm.LinearSVC()
model.fit(x_train_tfidf_norm, y_train)
predictions = model.predict(x_test_tfidf_norm)
print("------------------n-gram CountVektorizer SVC Sonuçları--------------")
print("Countvektorizer SVC ngram skor",accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print("--------------------------------------------------------")

#ANN MODEL

labels = ('Talep', 'Diğer','Şikayet')
colors = ['yellow','lightblue','green']

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

cat_ytrain = to_categorical(y_train, num_classes=3)
cat_ytest = to_categorical(y_test, num_classes=3)        
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
model1.add(Embedding(top_words, 64, input_length=num))
model1.add(Flatten())
model1.add(Dense(3,activation='softmax'))
model1.compile(loss='categorical_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])
model1.fit(X_train_seq_trunc,cat_ytrain, epochs=10, batch_size=16)
scores1=model1.evaluate(X_test_seq_trunc,cat_ytest)
score1 = model1.predict(X_test_seq_trunc)
print("-----------Result of Classical Artificial Neural Network--------------")
print("Accuracy: %.2f%%" % (scores1[1]*100))

cnn_y=argmax(score1,-1)
print(classification_report(y_test, cnn_y))
print(accuracy_score(y_test, cnn_y))
cnf_matrix = confusion_matrix(y_test, cnn_y)

# LSTM CLASSIFIER

model=Sequential()
model.add(Embedding(top_words, 256, input_length=num))
model.add(LSTM(16))
model.add(Dense(3, activation='softplus'))
model.compile(loss='categorical_crossentropy', 
             optimizer='adamax', 
             metrics=['accuracy'])
model.fit(X_train_seq_trunc,cat_ytrain, epochs=30,batch_size=64)
score1 = model.predict(X_test_seq_trunc)
lstm_y=argmax(score1,-1)
cm=confusion_matrix(y_test,lstm_y)
print("-----------Result of LSTM--------------")
print("LSTM Test= ",accuracy_score(y_test, lstm_y))
print(classification_report(y_test, lstm_y))