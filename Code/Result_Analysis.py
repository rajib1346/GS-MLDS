#!/usr/bin/env python
# coding: utf-8

# In[32]:


#Import library
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve

#load dataset
frame=pd.read_csv("E:/R3(DNA)/Material/R.chinensis/50_50/TP_TN_FP_FN_50.csv")
y_ture = frame[['Target']]
y_pre = frame[['Predict']]

#Calculate TN, TP, FP, FN from confusion_matrix
#Calculate Accuracy, Precision, TPR, FPR, TNR, FNR

TN, FP, FN, TP = confusion_matrix(y_ture, y_pre).ravel()
Acc = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)
TPR= TP/(TP+FN)
FPR = FP/(FP+TN)
TNR = TN / (TN+FP)
FNR = 1-TPR

print('Accuracy: ',Acc*100)
print('Precision: ',Precision*100)
print('TN: ',TN)
print('FP: ',FP)
print('FN: ',FN)
print('TP: ',TP)
print('TPR: ',TPR*100)
print('FPR: ',FPR*100)
print('TNR: ',TNR*100)
print('FNR: ',FNR*100)


# In[33]:


#ROC_AUC curve
fpr,tpr,treshold= roc_curve(y_ture,y_pre)
auc_sc=auc(fpr,tpr)*100
plt.figure(figsize=(5,5),dpi=100)
plt.plot([0,1],[0,1], color='orange',linestyle='--',label='model_auc (auc=%0.2f)' %auc_sc)
plt.plot(fpr,tpr,linestyle='-')
plt.xlabel('False Positive Rate-->')
plt.ylabel('True Positive Rate-->')
plt.legend()
plt.show()


# In[ ]:


#Area Under The Precision-Recall Curve (AUPRC)

train1=pd.read_csv("E:/R3(DNA)/Material/R.chinensis/50_50/Train_(60_40).csv")
test1=pd.read_csv("E:/R3(DNA)/Material/R.chinensis/50_50/Test(60_40).csv")
x_train1 =train1.drop(['Target'], axis=1)
y_train1 = train1[['Target']]
x_test1 =test1.drop(['Target'], axis=1)
y_test1 = test1[['Target']]

x_train1= [d.split() for d in train1['Gene Sequence'].tolist()]
x_test1= [d1.split() for d1 in test1['Gene Sequence'].tolist()]
train_tokenizer1 = Tokenizer()
train_tokenizer1.fit_on_texts(x_train1)
train_1= train_tokenizer1.texts_to_sequences(x_train1)
test_tokenizer1 = Tokenizer()
test_tokenizer1.fit_on_texts(x_test1)
test_1= test_tokenizer1.texts_to_sequences(x_test1)
train_max_length1 = len(x_train1[0])
train_X1=pad_sequences(train_1,maxlen=train_max_length1)
from tensorflow.keras.preprocessing.sequence import pad_sequences
test_max_length1 = len(x_test1[0])
test_X1=pad_sequences(test_1,maxlen=test_max_length1)

train_vocab_size1=len(train_tokenizer1.word_index) + 1
train_vocab1 = train_tokenizer1.word_index
test_vocab_size1=len(test_tokenizer1.word_index) + 1
test_vocab1 = test_tokenizer1.word_index
X_train1 = np.asarray(train_X1)
X_test1 = np.asarray(test_X1)
y_train1=to_categorical(y_train1)
y_test1=to_categorical(y_test1)

# create the embedding layer
embedding_layer1 = Embedding(train_vocab_size1,output_dim=DIM, weights=[embedding_vectors1], input_length=train_max_length1)
model = Sequential()
model.add(embedding_layer1)
model.add(Conv1D(64,8, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(X_train1,y_train1,epochs=10,batch_size=64, validation_data=(X_test1,y_test1))

y_score = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
auc_precision_recall = auc(recall, precision)
print(auc_precision_recall)

