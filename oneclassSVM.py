#!/usr/bin/env python
import json
import re
import os
import sys
import pickle
#import argparse
import numpy as np
import pandas as pd
from pprint import pprint
from konlpy.tag import Okt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 




data = pd.read_csv('./glowpick_review.csv',encoding='utf-8')
data = data.drop(["cmt_pk", "cmt_create_id",'time_update'], axis=1)
data = data.drop(['idx','channel_type'], axis=1)
data = data.drop(['time_create'],axis=1)


df = data[:10000]
#print(df)
df = df[['cmt_body','data_pk']]

dd = []
for i in range(10000):
    dd.append(1)

#print(dd)

df['label'] = dd

df = df.drop(['data_pk'],axis=1)

# random sampling
df = df.sample(frac=1)

# train data
train = df[:9500]
# print(train)

# glowpick test data
test = df[9500:]


#print(test['label'])


# ## other platform data (naver blog, youtube, powderroom, instagram)

data1 = pd.read_csv('./review_11.csv',encoding='utf-8')
data2 = pd.read_csv('./review_22.csv',encoding='utf-8')
data3 = pd.read_csv('./review_55.csv',encoding='utf-8')
data33 = pd.read_csv('./review_333.csv', encoding='utf-8')
data33 = data33[:789]
data33.rename( {'Unnamed: 1':'label'}, axis='columns', inplace=True)

data1 = data1[:500]
data2 = data2[:300]
data3 = data3[:150]
data33 = data33.drop(['keyword'],axis=1)



# concat
new33 = pd.concat([data1,data2,data3,data33])

new33 = new33.reindex(columns=['cmt_body','label'])
new33 = new33.dropna(axis=0)
new33 = new33.astype({'label': 'int64'})



# test data (glowpick + other platform)
test = pd.concat([test,new33])

# test data random sampling
test = test.sample(frac=1)


train['cmt_body']= train['cmt_body'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test['cmt_body']= test['cmt_body'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")



# x, y split
train_text = train['cmt_body'].tolist()
train_labels = train['label'].tolist()

test_text = test['cmt_body'].tolist()
test_labels = test['label'].tolist()


stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','넘']

test = test.dropna(subset=["cmt_body"])


# token
def tokenText(sample) :
    okt = Okt()
    v_text = []
    for sentence in sample:
        temp_b = []
        temp_b = okt.morphs(sentence, stem=True) # 토큰화
        temp_b = [word for word in temp_b if not word in stopwords] # 불용어 제거
        v_text.append(temp_b)
    tokens = [tok for tok in v_text]
    return tokens

train_tokens = tokenText(train_text)
test_tokens = tokenText(test_text)

# ### HashingVectorizer
#vectorizer = HashingVectorizer(n_features=300,tokenizer=tokenText)
#features = vectorizer.fit_transform(train_text).toarray()


# TfidfVectorizer


def corpus(tokens):
    corp = []
    for sent in tokens:
        tmp = " ".join(sent)
        corp.append(tmp)

    return corp

train_corp = corpus(train_tokens)
#print(train_corp)
test_corp = corpus(test_tokens)


def tfidfmatrix(corpus):
    vector = CountVectorizer(decode_error="replace")
    train_matrix = vector.fit_transform(train_corp)
    pickle.dump(vector.vocabulary_,open("feature.pkl","wb"))

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array(corpus)))

    return tfidf


train_matrix = tfidfmatrix(train_corp)
test_matrix = tfidfmatrix(test_corp)

#feat = train_matrix.toarray()



# oneclassSVM
# gridsearch

param={'kernel':['linear', 'rbf', 'poly'],
      'gamma':[.1, .01, .001, .0001],
      'nu':[0.25, 0.5, 0.75, 0.95]}
clf = GridSearchCV(estimator=OneClassSVM(),param_grid=param,cv=5, scoring = 'accuracy',n_jobs=-1)


clf.fit(train_matrix,train_labels)

# validate OneClassSVM model with train set
preds_train = clf.predict(train_matrix)
print("accuracy:", accuracy_score(train_labels, preds_train))

# validate OneClassSVM model with test set
preds_test = clf.predict(test_matrix)
#preds_test

results = confusion_matrix(test_labels, preds_test) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(test_labels, preds_test)) 
print('Report : ')
print(classification_report(test_labels, preds_test))

roc = roc_auc_score(test_labels, preds_test)
print(f'ROC score is {roc}')

