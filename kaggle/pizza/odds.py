# http://www.kaggle.com/c/random-acts-of-pizza/data?sampleSubmission.csv.zip

import pandas as pd
from sklearn import ensemble
import math,sys,resource
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from string import punctuation




loc_train = "data/train.json"
min_word_length = 3 # for words in odds ratio
number_of_top_words = 200 # words with highest counts

df_train = pd.read_json(loc_train)

feature_col = 'request_text'
feature_col = 'request_title'
result_col = 'requester_received_pizza'
get_columns = [feature_col,result_col]
feature_cols = [col for col in df_train.columns if col in get_columns]
print "feature_cols:"
print feature_cols
features = df_train[feature_cols]
good_words = {}
failed_words = {}

for i in features.iterrows():
	feature_info = i[1]
	request_text = feature_info['request_title']
	result = feature_info['requester_received_pizza']
	if result :   # got pizza!
        	for word in request_text.strip().split():
            		word = word.strip(punctuation).lower()
			word = word.encode('ascii','ignore')
			word.replace("'",'')
            		if ( word not in stop_words and (len(word) >= min_word_length) and word.isalpha() ) :
				#print "Found word: [%s]" % word
				if word in good_words.keys() :
					good_words[word] += 1
				else :
					good_words[word] = 1


	else :   # sorry no pizza
        	for word in request_text.strip().split():
            		word = word.strip(punctuation).lower()
			word = word.encode('ascii','ignore')
			word.replace("'",'')
            		if ( word not in stop_words and (len(word) >= min_word_length) and word.isalpha() ) :
				#print "Found word: [%s]" % word
				if word in failed_words.keys() :
					failed_words[word] += 1
				else :
					failed_words[word] = 1





top_words = sorted(good_words.iteritems(), key=lambda(word, count): (-count, word))[:number_of_top_words]

print "The ", number_of_top_words , "Most Frequently Used Words: "

odds_data = {}
for word, counts in top_words:
    failed_counts = failed_words[word]
    odds = (counts)/(failed_counts+0.1)
    odds_data[word] = odds
    print "%s: %d %d %.3f" % (word, counts, failed_counts,odds)

number_of_top_odds_words = number_of_top_words//5
top_odds = sorted(odds_data.iteritems(), key=lambda(word, count): (-count, word))[:number_of_top_odds_words]

print "==================================================="
print "The ", number_of_top_words , "Top Odds Words: "

odds_data = {}
for word, odds in top_odds:
    print "%s: %.3f" % (word, odds)


