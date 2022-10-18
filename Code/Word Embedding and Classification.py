"""After each layer's classification is complete, the correctly classified data is added to the pre-training data 
and compared with the predefined target values in the original data set and inserted into the next layer. In the next 
iteration, the misclassified data is used as new test data. This process continues until all models have improved 
performance (TP and TN = 0)"""

#Lort library
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import gensim
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.layers import Conv1D,GlobalMaxPooling1D, Flatten
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load data
train1=pd.read_csv("E:/R3(DNA)/Material/R.chinensis/50_50/1st_iteration/Train.csv")
test1=pd.read_csv("E:/R3(DNA)/Material/R.chinensis/50_50/1st_iteration/Test.csv")

x_train1 =train1.drop(['Target'], axis=1)
y_train1 = train1[['Target']]
x_test1 =test1.drop(['Target'], axis=1)
y_test1 = test1[['Target']]

#Count total word
x_train1= [d.split() for d in train1['Gene Sequence'].tolist()]
x_test1= [d1.split() for d1 in test1['Gene Sequence'].tolist()]

#Tokenize every word
train_tokenizer1 = Tokenizer()
train_tokenizer1.fit_on_texts(x_train1)
train_1= train_tokenizer1.texts_to_sequences(x_train1)
test_tokenizer1 = Tokenizer()
test_tokenizer1.fit_on_texts(x_test1)
test_1= test_tokenizer1.texts_to_sequences(x_test1)

#Padding the sequence
train_max_length1 = len(x_train1[0])
train_X1=pad_sequences(train_1,maxlen=train_max_length1)
test_max_length1 = len(x_test1[0])
test_X1=pad_sequences(test_1,maxlen=test_max_length1)

#Count the vocabulary size
train_vocab_size1=len(train_tokenizer1.word_index) + 1
train_vocab1 = train_tokenizer1.word_index
test_vocab_size1=len(test_tokenizer1.word_index) + 1
test_vocab1 = test_tokenizer1.word_index

print('Total word (Train): ', len(x_train1[0]))
print('Total word (Test): ', len(x_test1[0]))
print('\nToken (Train): ', train_tokenizer1.word_index)
print('\nToken (Test): ', test_tokenizer1.word_index)
print('\nPading (Train[0]): ', train_X1[0])
print('Padding (Test[0]): ', test_X1[0])
print('\nVocab Size (Train): ', len(train_vocab1))
print('Vocab Size (Test): ', len(test_vocab1))

#Transfer the sequences into vector form
DIM=100
def get_weight_matrix1(model):
    weight_matrix = np.zeros((train_vocab_size1, DIM))
    
    for word, i in train_vocab1.items():
        weight_matrix[i]=model.wv[word]
    return weight_matrix

w2v_model1 = gensim.models.Word2Vec(sentences=x_train1, size=100, window=10, min_count=5, negative=5, epochs=25)
embedding_vectors1 = get_weight_matrix1(w2v_model1)

#Converted into a matrix 
X_train1 = np.asarray(train_X1)
X_test1 = np.asarray(test_X1)
y_train1=to_categorical(y_train1)
y_test1=to_categorical(y_test1)

#Create the embedding layer
embedding_layer1 = Embedding(train_vocab_size1,output_dim=DIM, weights=[embedding_vectors1], input_length=train_max_length1)

#Build the model
#(Model_1)
model1 = Sequential()
model1.add(embedding_layer1)
model1.add(Conv1D(64,8, activation='relu'))
model1.add(MaxPooling1D(2))
model1.add(Dropout(0.5))
model1.add(Dense(32, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(16, activation='relu'))
model1.add(GlobalMaxPooling1D())
model1.add(Dense(2, activation='softmax'))
model1.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
model1.summary()
model1.fit(X_train1,y_train1,epochs=10,batch_size=64, validation_data=(X_test1,y_test1))

#(Model_2)
model2 = Sequential()
model2.add(embedding_layer1)
model2.add(Conv1D(64,8, activation='relu'))
model2.add(MaxPooling1D(2))
model2.add(Dropout(0.5))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.8))
model2.add(Dense(16, activation='relu'))
model2.add(GlobalMaxPooling1D())
model2.add(Dense(2, activation='softmax'))
model2.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
model2.summary()
model2.fit(X_train1,y_train1,epochs=10,batch_size=64, validation_data=(X_test1,y_test1))

#(Model_3)
model3 = Sequential()
model3.add(embedding_layer1)
model3.add(Conv1D(64,8, activation='relu'))
model3.add(MaxPooling1D(2))
model3.add(Dropout(0.5))
model3.add(Dense(32, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(16, activation='relu'))
model3.add(GlobalMaxPooling1D())
model3.add(Dense(2, activation='softmax'))
model3.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
model3.summary()
model3.fit(X_train1,y_train1,epochs=10,batch_size=64, validation_data=(X_test1,y_test1))

#Grid search for the best combination of w1, w2, w3 that gives maximum acuracy
models1 = [model1, model2, model3]
preds1 = [model.predict(X_test1) for model in models1]
preds1=np.array(preds1)

y_actual1 = pd.DataFrame(y_test1)
y_actual1 = pd.DataFrame(y_actual1.idxmax(axis=1))
y_actual1 = np.asarray(y_actual1)

df1 = pd.DataFrame([])

for w1 in range(0, 5):
    for w2 in range(0,5):
        for w3 in range(0,5):
            wts = [w1/10.,w2/10.,w3/10.]
            wted_preds1 = np.tensordot(preds1, wts, axes=((0),(0)))
            wted_ensemble_pred1 = np.argmax(wted_preds1, axis=1)
            weighted_accuracy1 = accuracy_score(y_actual1, wted_ensemble_pred1)
            df1 = df1.append(pd.DataFrame({'wt_a':wts[0],'wt_b':wts[1], 
                                         'wt_c':wts[2], 'acc':weighted_accuracy1*100}, index=[0]), ignore_index=True)
max_acc_row1 = df1.iloc[df1['acc'].idxmax()]
print(max_acc_row1)

#Average Ensemble
summed1 = np.sum(preds1, axis=0)
AVE_prediction1 = np.argmax(summed1, axis=1)

a_prediction1 = model1.predict_classes(X_test1)
b_prediction1 = model2.predict_classes(X_test1)
c_prediction1 = model3.predict_classes(X_test1)

a_accuracy1 = accuracy_score(y_actual1, a_prediction1)
b_accuracy1 = accuracy_score(y_actual1, b_prediction1)
c_accuracy1 = accuracy_score(y_actual1, c_prediction1)
AVE_accuracy1 = accuracy_score(y_actual1, AVE_prediction1)

#Weaighted Average Ensemble
weights1 = [0.2, 0.1, 0.4]
weighted_preds1 = np.tensordot(preds1, weights1, axes=((0),(0)))
WAE_prediction1 = np.argmax(weighted_preds1, axis=1)
WAE_accuracy1 = accuracy_score(y_actual1, WAE_prediction1)

print('Accuracy Score for model1 = ', a_accuracy1)
print('Accuracy Score for model2 = ', b_accuracy1)
print('Accuracy Score for model3 = ', c_accuracy1)
print('Accuracy Score for average ensemble = ', AVE_accuracy1)
print('Accuracy Score for weighted average ensemble = ', WAE_accuracy1)

#Separate correctly classified and misclassified data and save in different csv files
print("Train_size:",len(train1),"| Test_size:",len(test1))

if AVE_accuracy1 > WAE_accuracy1 :
	test1['AVE']=AVE_prediction1
	test1['match'] = np.where(test1['Target'] == test1['AVE'], 'True', 'False')
	FP_FN_AVE_1=test1[test1['match']=='False']
	FP_FN_AVE_1=FP_FN_AVE_1.drop(['AVE','match'], axis=1)
	FP_FN_AVE_1.to_csv('E:/R3(DNA)/Material/R.chinensis/50_50/2nd_iteration/Test_2.CSV')
	print("AVE_FP_FN: ",len(FP_FN_AVE_1))

	TP_TN_AVE_1=test1[test1['match']=='True']
	TP_TN_AVE_1.to_csv('E:/R3(DNA)/Material/R.chinensis/50_50/1st_iteration/1.CSV')
	print("AVE_TN_TP: ",len(TP_TN_AVE_1))


elif AVE_accuracy1 < WAE_accuracy1 :
	test1['WAE']= WAE_prediction1
	test1['match'] = np.where(test1['Target'] == test1['WAE'], 'True', 'False')
	FP_FN_WAE_1=test1[test1['match']=='False']
	FP_FN_WAE_1=FP_FN_WAE_1.drop(['WAE','match'], axis=1)
	FP_FN_WAE_1.to_csv('E:/R3(DNA)/Material/R.chinensis/50_50/2nd_iteration/Test_2.CSV')
	print("WAE_FP_FN: ",len(FP_FN_WAE_1))

	TP_TN_WAE_1=test1[test1['match']=='True']
	TP_TN_WAE_1.to_csv('E:/R3(DNA)/Material/R.chinensis/50_50/1st_iteration/1.CSV')
	print("WAE_TN_TP: ",len(TP_TN_WAE_1))
    
else:
	test1['WAE']= WAE_prediction1
	test1['match'] = np.where(test1['Target'] == test1['WAE'], 'True', 'False')
	FP_FN_WAE_1=test1[test1['match']=='False']
	FP_FN_WAE_1=FP_FN_WAE_1.drop(['WAE','match'], axis=1)
	FP_FN_WAE_1.to_csv('E:/R3(DNA)/Material/R.chinensis/50_50/2nd_iteration/Test_2.CSV')
	print("WAE_FP_FN: ",len(FP_FN_WAE_1))

	TP_TN_WAE_1=test1[test1['match']=='True']
	TP_TN_WAE_1.to_csv('E:/R3(DNA)/Material/R.chinensis/50_50/1st_iteration/1.CSV')
	print("WAE_TN_TP: ",len(TP_TN_WAE_1))
