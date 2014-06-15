

# The following 12 request_text words had the highest odds for success in getting pizza
#rice: 0.858
#person: 0.773
#edit: 0.695
#paycheck: 0.686
#went: 0.662
#request: 0.657
#wife: 0.642
#currently: 0.641
#start: 0.618
#offer: 0.615
#check: 0.609
#recently: 0.604
#  add feature for each and sum of feature existences for all

# The following has the best odds for request_title
#cooking: 1.935
#anniversary: 1.373
#asking: 1.373
#ask: 1.176
#christmas: 1.176
#happy: 1.111
#baby: 0.986
#father: 0.986
#arizona: 0.891
#daughter: 0.891
#kentucky: 0.864
#yesterday: 0.864
#laid: 0.840
#late: 0.826


import pandas as pd
from string import punctuation
import numpy as np
import sys

def feature_engineer_text(df,top_words,raw_feature_column):
	total_name = 'total_top_words' + '_' + raw_feature_column
	top_words_plus_total = ['row_number'] + top_words + [total_name]
	print "Input shape: ", df.shape
	df_shape = df.shape
	x = df_shape[0]
	column_max = len(top_words)
	#new_features = np.zeros((x,column_max+1)) # will populate with 1's for hits
	new_features = pd.DataFrame(index=range(x),columns=top_words_plus_total)
	new_features.fillna(0,inplace=1) # fill with zeros
	print "New Features shape:", new_features.shape
	#print new_features
	feature_col = raw_feature_column
	get_columns = ['request_id',feature_col]  # request_id used for joining
	feature_cols = [col for col in df.columns if col in get_columns]
	features = df[feature_cols]

	row_number = 0
	prev_found_row = 0
	found_row = 0
	for i in features.iterrows():
        	feature_info = i[1]
        	request_text = feature_info[raw_feature_column]
        	request_id = feature_info['request_id']
		new_features.loc[row_number,'request_id'] = request_id
		# now create additional features based on this
        	for word in request_text.strip().split():
			word = word.strip(punctuation).lower()
                	word = word.encode('ascii','ignore')
                	word.replace("'",'')
			if word in top_words :
				column = top_words.index(word)
				#print row_number, column, ", Found word: [",word,"], in [",request_text,"]"
				new_features.loc[row_number,word] = 1
				#print "nfw:",new_features.loc[row_number,word] 
				#sys.exit()
				new_features.loc[row_number,total_name] += 1 # grand totals
				new_features.loc[row_number,'row_number'] = row_number
				prev_found_row = found_row  # for debugging
				found_row = row_number
				

				
		row_number += 1

	# debugging
	#print "NEW FEATURES:"
	#print new_features
	#print row_number,column_max
	#print "DF columns:", df.columns
	new_features = new_features.drop('row_number',1)  # dont keep this column anymore
	new_features = new_features.drop('request_id',1)  # dont keep this column anymore
	#print "NF columns:", new_features.columns
	#res = pd.concat((df,new_features),join_axes='request_id')
	res = pd.concat((df,new_features),1)
	#print "SHAPE RES: ",res.shape
        #print "RES COLUMNs:", res.columns

	#print y[prev_found_row,...]
	#print y[found_row,...]
	# finally concatonate new_features in df and return



	return res
