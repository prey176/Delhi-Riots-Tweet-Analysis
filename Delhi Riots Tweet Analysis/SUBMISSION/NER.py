import nltk
import pickle
import string
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
def load():
	dbfile = open('tweets', 'rb')
	db = pickle.load(dbfile)
	dbfile.close()
	return db

tweets_text = load()
namedEntities = {}

for tweet_text in tweets_text:
	tokens = nltk.word_tokenize(tweet_text)
	tags = nltk.pos_tag(tokens)
	chunk = nltk.ne_chunk(tags)
	NE = [ " ".join(w for w,t in ele) for ele in chunk if isinstance(ele, nltk.Tree)]
	for entity in NE :
		if entity not in namedEntities : 
			namedEntities[entity] = 0
		namedEntities[entity] += 1 

list_ = []
for entity in namedEntities :
	list_.append((entity,namedEntities[entity]))

list_.sort(key = lambda x : x[1], reverse = True)

for a in list_ :
	print (a)