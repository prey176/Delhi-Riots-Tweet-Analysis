#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
import os
import tweepy
import pickle
from sklearn.naive_bayes import MultinomialNB


# In[2]:


#Consumer Key (API Key), Consumer Secret (API Secret)
auth = tweepy.OAuthHandler('s9iqHQMCnqbRSTYLZijznltjj', 'edpnKGn7l90SPKoFBc5eiBi2kEjb5sFe5CfH2vZ4O51g1lvfXw')
# Access Token, Access Token Secret
auth.set_access_token('914187849875914752-qyp1TmXtXp4BDUOzknUlPdF9owp582g', '7SpLBABZOAnph1j1AmA6AYodpdWdRx0IXHaWQ7muTwQKM')
api = tweepy.API(auth)
if (not api):
    print("Authentication failed :(")
else:
    print("Authentication successfull!!! :D")


# In[3]:


def getmodel():
    dbfile = open('model', 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    return db


# In[4]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(sentence,lan):
    if lan=='en':
        vs = analyzer.polarity_scores(sentence)
        return vs['compound']
    else:
        return None


# In[5]:


def feature_list(url) :
    # url will be of the form https://twitter.com/handle/status/tweet_id
    
    handle = url.split('/')[-3]
    id_ = int(url.split('/')[-1])
    tweet = api.get_status(id_,tweet_mode='extended')
    row_new = [0 for i in range(9)]
    check = 0
    user = api.get_user(screen_name=handle)
    if user.verified:
        row_new[0]=1

    if user.location is not None:
        row_new[1]=1

    if user.name is not None or user.name!="":
        row_new[2]=1

    if "profile_image_url_https" in user.profile_image_url_https!="":
        row_new[3]=1

    row_new[4] = get_sentiment(user.description,"en")
    row_new[5] = tweet.favorite_count
    row_new[6] = tweet.retweet_count
    row_new[7] = get_sentiment(tweet.full_text,"en")
    
    featureSet_test = np.array([row_new])
    featureSet_test = np.asarray(featureSet_test, dtype='float64')
    
    return featureSet_test


# In[9]:


def solve(url = 'https://twitter.com/PettyPraxis/status/1232793544860979201') :
    model = getmodel()
    featureSet_test = feature_list(url)
    predict = model.predict(featureSet_test)
    if (predict == '0') :
        return 'Not Rumour'
    else :
        return 'Rumour'

