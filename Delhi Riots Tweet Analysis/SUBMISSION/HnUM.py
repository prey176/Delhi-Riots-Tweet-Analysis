import nltk
import pickle
import re
def load():
	dbfile = open('tweets', 'rb')
	db = pickle.load(dbfile)
	dbfile.close()
	return list(set(db))

tweets_text = load()

print ('Hashtags')
# another possible regex for hastag -> (?:^|\s)[＃#]{1}(\w+)
for tweet_text in tweets_text :
	a = re.findall(r"\B(\#[a-zA-Z]+\b)",tweet_text)
	print(*a)

print ('Usermentions')
# another possible regex for usermention -> (?:^|\s)[@@]{1}(\w+)
for tweet_text in tweets_text :
	a = re.findall(r"\B(\@[a-zA-Z]+\b)",tweet_text)
	print(*a)
    
print ('Mobile Numbers')
for tweet_text in tweets_text :
	a = re.findall(r"^((\+){1}91){1}[1-9]{1}[0-9]{9}$",tweet_text)
	print(*a)

print("Vehicle Numbers")
pattern = "([A-Z]{2}[ -][0-9]{1,2}(?:[A-Z])?(?:[ -][A-Z])?[ -][0-9]{4})"
for tweet in tweets_text:
    a = re.findall(pattern, tweet)
    print(*a)