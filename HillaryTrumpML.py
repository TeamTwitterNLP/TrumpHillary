#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

tweets="data/file_name.csv"
tweets_df = pd.read_csv(tweets)
tweets_df.head()


# In[2]:


#only take handle and text columns
new_df=tweets_df[['handle','text']]

new_df.head()


# In[3]:


#convert handle into label 0 or 1
mapping = {'realDonaldTrump': 0, 'HillaryClinton': 1}

new_df2=new_df.replace({'handle': mapping})

new_df2.head()


# In[4]:


#drop any null colomuns
new_df2drop=new_df2.dropna()


# In[5]:


#use below code to download and install NLTK package

#import nltk
#nltk.download()


# In[6]:


#remove stopwords
from nltk.corpus import stopwords

def no_user_alpha(tweet):
        clean_mess = [word for word in tweet.split() if word.lower() not in stopwords.words('english')]
        return clean_mess

   

print(no_user_alpha(new_df['text'].iloc[10]))


# In[7]:



import sys
import argparse
import logging
from logging import critical, error, info, warning, debug

import numpy as np
import seaborn as sn
import pandas as pd
import random

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split


# In[8]:


X = new_df2drop['text']
y = new_df2drop['handle']

y


# In[9]:


msg_train, msg_test, label_train, label_test = train_test_split(new_df2drop['text'], new_df2drop['handle'], test_size=0.2)


# In[10]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=no_user_alpha)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(msg_train,label_train)


# In[11]:



predictions = pipeline.predict(msg_test)

print(classification_report(predictions,label_test))
print ('\n')
print(confusion_matrix(predictions,label_test))
print(accuracy_score(predictions,label_test))


# In[118]:


newtweet2="Libya"

newtweet = {'handle':  [0],
        'text': newtweet2}

newtweetdf= pd.DataFrame (newtweet, columns = ['handle','text'])

newtweetdf


# In[119]:



predictions = pipeline.predict(newtweetdf['text'])
#print(confusion_matrix(predictions,label_test))
probability = pipeline.predict_proba(newtweetdf['text'])
TrumpHillaryScore=predictions[0]

print(TrumpHillaryScore)

#print(accuracy_score(predictions,1))
print(probability)


# In[ ]:




