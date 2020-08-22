#!/usr/bin/env python
# coding: utf-8

# # DataSet - Training

# In[62]:


import csv
import numpy as np


dataset = []
index = 0
fields = ['verified', 'profile_location', 'profile_name', 'profile_image', 'account_date', 'profile_desc', 'fav_count', 'retweet_count', 'tweet_location', 'text', 'media', 'rumour_label']
with open("train_data.csv",'r') as file:
    file = csv.reader(file)
    header = next(file)
    for row in file:
        dataset.append(row)

dataset = np.array(dataset)
np.random.seed()
np.random.shuffle(dataset)
print(dataset.shape)


# # Authentication - Twitter Api

# In[63]:


import os
import tweepy

auth = tweepy.OAuthHandler('s9iqHQMCnqbRSTYLZijznltjj', 'edpnKGn7l90SPKoFBc5eiBi2kEjb5sFe5CfH2vZ4O51g1lvfXw')
# Access Token, Access Token Secret
auth.set_access_token('914187849875914752-qyp1TmXtXp4BDUOzknUlPdF9owp582g', '7SpLBABZOAnph1j1AmA6AYodpdWdRx0IXHaWQ7muTwQKM')
api = tweepy.API(auth)

if (not api):
    print("Authentication failed :(")
else:
    print("Authentication successfull!!! :D")


# # Extracting features and labels - Training Set

# In[64]:


numFeatures = dataset.shape[1]-1
numSamples = dataset.shape[0]

labels = dataset[:,-1]
featureSet = np.delete(dataset,numFeatures,1)
featureSet = np.delete(featureSet,8,1)
featureSet = np.delete(featureSet,4,1)
print(featureSet.shape)


# In[65]:


ee1 = featureSet[:800]
ee2 = labels[:800]
print (ee1.shape)
print (ee2.shape)


# In[66]:


fields = ['verified', 'profile_location', 'profile_name', 'profile_image', 'profile_desc', 'fav_count', 'retweet_count', 'text', 'media',]


# In[67]:


print(labels.shape)


# # Sentiment Analyzer

# In[68]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(sentence,lan):
    if lan=='en':
        vs = analyzer.polarity_scores(sentence)
        return vs['compound']
    else:
        return None


# In[69]:


print(get_sentiment("Bloody hell ! my friends towards #Durgapuri are all stuck in jam for hours now . Enough of cowardice @DelhiPolice","en"))


# In[70]:


print(featureSet[0])


# # Preprocessing of Training Set

# In[71]:


for sample in ee1:
    for i in range(len(fields)):
        if(fields[i]=="verified" or fields[i]=="media"):
            if(sample[i]=="True"):
                sample[i] = int(1)
            else:
                sample[i] = int(0)
        
        elif(fields[i]=="profile_location" or fields[i]=="profile_name" or fields[i]=="profile_image"):
            if(sample[i]!="" and sample[i]!=" "):
                sample[i]=int(1)
            else:
                sample[i]=int(0)
                
        elif(fields[i]=="profile_desc" or fields[i]=="text"):
            sample[i] = get_sentiment(sample[i],"en")
            
        else:
            sample[i] = int(sample[i])


# In[72]:


for i in range(labels.shape[0]):
    if(labels[i]=="True"):
        labels[i]=1
    else:
        labels[i]=0


# In[73]:


labels = labels.astype(np.float64)
ee1 = ee1.astype(np.float64)
ee2 = ee2.astype(np.float64)


# # Multinomial Naive Bayes

# In[74]:


from sklearn.naive_bayes import MultinomialNB
ee1 = np.abs(ee1)
gnb = MultinomialNB()
gnb.fit(ee1,ee2)


# In[135]:


print("Accuracy on the training set:")
print(gnb.score(ee1,ee2))


# # Random Forest Classifier

# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[26]:


clf = RandomForestClassifier()
clf.fit(ee1,ee2)


# In[28]:


print("Accuracy on the Training Set:")
print(clf.score(ee1,ee2))


# # Gaussian Naive Bayes 

# In[50]:


from sklearn.naive_bayes import GaussianNB
clf2 = GaussianNB()
clf2.fit(ee1,ee2)


# In[51]:


print("Accuracy on Training Set:")
print(clf2.score(ee1,ee2))


# # DataSet , Preprocessing - Test

# In[75]:


featureSet_test = []
labels_test = []
with open("final_tweets.csv","r") as file:
    file = csv.reader(file)
    header = next(file)
        
    check = 0
    count = 0
    ids = {}
    for row in file:
        row_new = [0 for i in range(9)]
        check = 0
        handle = row[1]
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
        row_new[5] = row[5]
        row_new[6] = row[4]
        row_new[7] = get_sentiment(row[6],"en")
        
            
        permalink = row[11]
        
        print(row[16])
        if(row[16]=="Rumour"):
            print("entered")
            check = 1
        
        featureSet_test.append(row_new)
        labels_test.append(check)
        cur_id = (row[11].split('/')[-1])
        print(cur_id)
        ids[cur_id] = count
        count += 1
        
featureSet_test = np.array(featureSet_test)
featureSet_test = np.asarray(featureSet_test, dtype='float64')
labels_test = np.array(labels_test)

print(len(ids))
print(featureSet_test.shape)
# np.reshape(labels_test,(featureSet_test.shape[0],1))
# prnit(featureSet_test.shape)


# # Loading the Saved Model

# In[76]:


import pickle
def load():
    dbfile = open('model', 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    return db
model = load()


# # Multinomial Naive Bayes

# In[77]:


import numpy as np
featureSet_test = np.asarray(featureSet_test, dtype='float64')
labels_test = np.asarray(labels_test, dtype='float64')
q = model.predict(featureSet_test)
q = np.asarray(q, dtype='float64')


# # Accuracy

# In[78]:


def accuracy(a,b):
    return np.sum(a==b)/(b.shape[0])


# In[79]:


print("Accuracy on Test Set:")
print(accuracy(q,labels_test))
# print(model.score(featureSet_test,labels_test))


# # True Positives

# In[81]:


def check(predicted):
    count = 0 
    pp = 0
    for i in range (len(predicted)) :
        if predicted[i] == labels_test[i] and predicted[i] > 0 :
            count+=1
        if labels_test[i] == 1.0:
            pp+=1
    print (count)
    print (pp)
check(q)


# # Confusion Matrix

# In[82]:


from sklearn.metrics import confusion_matrix
print("Confusion Matrix for the model:")
print(confusion_matrix(labels_test,q))


# # ROC Curve

# In[83]:


from sklearn.metrics import roc_curve, auc
print(ee2.shape)
y_score = model.predict_proba(featureSet_test)
print(y_score.shape)
fpr, tpr, _ = roc_curve(labels_test,y_score[:,1])
roc_auc = auc(fpr, tpr)


# In[84]:


import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# # Random Forest Classifier 

# In[85]:


print("Accuracy on Test Set:")
print(clf.score(featureSet_test,labels_test))
pred2 = clf.predict(featureSet_test)
check(pred2)


# # Gaussian Naive Bayes

# In[52]:


print("Accuracy on Test Set:")
print(clf2.score(featureSet_test,labels_test))
pred3 = clf2.predict(featureSet_test)
check(pred3)


# # Predictions of Multinomial Naive Bayes

# In[27]:


for i in range(len(q)):
    print ('Row Number', i+2, 'Predicted',q[i],'Actual',labels_test[i])


# # Rumour Spread Function

# In[86]:


def rumour_spread(id):
    id = str(id)
    cur_feature = featureSet_test[ids[id]]
    print(model.predict_proba(cur_feature.reshape(1, -1)))


# In[87]:


print(rumour_spread(1229271298880217090))

