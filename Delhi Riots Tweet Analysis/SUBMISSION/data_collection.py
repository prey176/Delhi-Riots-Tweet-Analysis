#!/usr/bin/env python
# coding: utf-8

# In[1]:


import GetOldTweets3 as got

accounts = ["@DelhiPolice ", "@CPDelhi ", "@DCPSouthDelhi ", "@DCPEastDelhi ", "@DCPSEastDelhi ", "@DCPNEastDelhi ", "@DcpNorthDelhi ", "@DCPCentralDelhi ", "@DCPNewDelhi "]
locations = ['Durgapuri','durgapuri','brijpuri','Brijpuri','DURGAPURI', 'BRIJPURI']

tweets = []
for acc in accounts:
    for loc in locations:
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(acc + loc)                                               .setSince("2020-02-01")                                               .setUntil("2020-03-06")                                               .setMaxTweets(1000)
        tweets += got.manager.TweetManager.getTweets(tweetCriteria)
        print(len(tweets))
print(len(tweets))


# In[7]:


outputFile = open("temp_tweets.csv", "w+", encoding="utf8")
outputFile.write('date,username,to,replies,retweets,favorites,text,geo,mentions,hashtags,id,permalink\n')
import json

for t in tweets:
    y = str(t)
    print(type(t))
    data = [t.date.strftime("%Y-%m-%d %H:%M:%S"),
            t.username,
            t.to or '',
            t.replies,
            t.retweets,
            t.favorites,
            '"'+t.text.replace('"','""')+'"',
            t.geo,
            t.mentions,
            t.hashtags,
            t.id,
            t.permalink]
    data[:] = [i if isinstance(i, str) else str(i) for i in data]
    outputFile.write(','.join(data) + '\n')
outputFile.flush()


# In[ ]:




