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
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Real Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

data1=pd.read_csv('iki_data.csv')
data1=data1.drop(columns=['Unnamed: 0'])

train=[]
train1=[]

notr=data1[data1['1']=='Nötr' ]
pozitif=data1[data1['1']=='Pozitif' ]
negatif=data1[data1['1']=='Negatif' ]


frame1 = [notr,pozitif,negatif]
df = pd.concat(frame1)
cizim=df[df.columns[1]].value_counts()
train=[]
y=[]

for i in df['1']:
    if i=='Negatif':
        y.append(0)
    elif i=='Nötr':
        y.append(1)
    elif i=='Pozitif':
        y.append(2)
        
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

labels = ('Negative', 'Neutral','Positive')
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



# LSTM CLASSIFIER

model=Sequential()
model.add(Embedding(top_words, 256, input_length=num))
model.add(LSTM(16))
model.add(Dense(5, activation='softplus'))
model.compile(loss='categorical_crossentropy', 
             optimizer='adamax', 
             metrics=['accuracy'])
history=model.fit(X_train_seq_trunc,cat_ytrain, epochs=10,batch_size=512)
score1 = model.predict(X_test_seq_trunc)
lstm_y=argmax(score1,-1)
cm=confusion_matrix(y_test,lstm_y)
print("-----------Result of LSTM--------------")
print("LSTM Test= ",accuracy_score(y_test, lstm_y))
print(classification_report(y_test, lstm_y))

np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm, classes=labels,
                      title='Confusion matrix')

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
