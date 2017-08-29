dev_mode=True # use this to run on subset of training while developing this script

import numpy as np 
import pandas as pd 
from collections import Counter
import pandas as pd
import numpy as np
import re
import gc; gc.enable()
from sklearn.feature_extraction import text
from sklearn import naive_bayes
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def smape_fast(y_true, y_pred):
       y_true=pd.DataFrame(y_true)
       y_pred=pd.DataFrame(y_pred)
       assert y_true.shape[1]==1
       assert y_pred.shape[1]==1
       df=pd.concat([y_true, y_pred], axis=1)
       df.columns=['true', 'pred']
       df['sum']=df['true']+df['pred']
       df['diff']=df['true']-df['pred']
       df['diff']=pd.DataFrame.abs(df['diff'])
       df['smape_base']=df['diff']/df['sum']
       out=df['smape_base'].sum()
       out*= (200/y_true.shape[0])
       return out

def get_lang_from_page(page): 
    lang = 'ts' # just throw into this bucket since later in the script it tries to reprocess these cases which arise from bogus splitting.
    if re.search(r"\/[a-z]{2}_www.mediawiki.org", page):
        x = re.split("_www.mediawiki.org",page)
        lang = re.split("/",x[0])[-1]
        print("MEDIAWIKI www: [",page,"], lang:", lang)
    elif re.search(r"\/[a-z]{2}_commons.wikimedia.org", page):
        x = re.split("_www.mediawiki.org",page)
        lang0 = re.split("/",x[0])[-1]
        lang = re.split("_",lang0)[0] 
        print("WIKIMEDIA commons: [",page,"], lang:", lang)
    elif page.find(r'wikipedia.org')>0:
        lang = re.split(".wikipedia.org", page)[0][-2:]
    else :
        print("NOMATCH:", page)
    return lang

train = pd.read_csv("../input/train_1.csv")
#determine idiom with URL
#train['language']=train['Page'].apply(lambda x:re.split(".wikipedia.org", x)[0][-2:])
train['language']=train['Page'].apply(get_lang_from_page)
'''
This is what you get with a value counts on train.language
en    24108
ja    20431
de    18547
fr    17802
zh    17229
ru    15022
es    14069
ts    13556
er     4299
'''

print("Counter train language:", Counter(train.language))


print("Train Describe:", train.describe())
print("Train head:", train.head())
print("Train shape:", train.shape)


#we have english, japanese, deutch, french, chinese (taiwanese ?), russian, spanish
#ts and er are cases where language could not be determined as part of the page name
# try to replace them by learning from special chars
#Note : this step wasn't tuned, and can't be perfect because other idioms are available in those Pages (such as portuguese for example)

#let's make a train, target, and test to predict language on ts and er pages
orig_train=train.loc[~train.language.isin(['ts', 'er']), 'Page']
orig_target=train.loc[~train.language.isin(['ts', 'er']), 'language']
orig_test=train.loc[train.language.isin(['ts', 'er']), 'Page']
#keep only interesting chars
orig_train2=orig_train.apply(lambda x:x.split(".wikipedia")[0][:-3]).apply(lambda x:re.sub("[a-zA-Z0-9():\-_ \'\.\/]", "", x))
orig_test2=orig_test.apply(lambda x:x.split(".wikipedia")[0][:-3]).apply(lambda x:re.sub("[a-zA-Z0-9():\-_ \'\.\/]", "", x))
print("Orig Train2:")
print(orig_train2.describe())
print(orig_train2.head())

#run TFIDF on those specific chars
tfidf=text.TfidfVectorizer(input='content', encoding='utf-8', decode_error='strict', strip_accents=None, 
                     lowercase=True, preprocessor=None, tokenizer=None, 
                     analyzer='char', #stop_words=[chr(x) for x in range(97,123)]+[chr(x) for x in range(65,91)]+['_','.',':'], 
                     token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=True, norm='l2', 
                     use_idf=True, smooth_idf=True, sublinear_tf=False)
orig_train2a = orig_train2
orig_train2=tfidf.fit_transform(orig_train2)
#model=naive_bayes.BernoulliNB()
n_estimators=100
n_estimators=5
n_estimators=200
max_features=3000
model=RandomForestClassifier(n_estimators=n_estimators)


model.fit(orig_train2, orig_target)
result_on_training = model.predict(tfidf.transform( orig_train2a ) )
print("orig_train2 after fit:", orig_train2)
print("orig_train2 after fit:", type(orig_train2))
print("orig target:", orig_target)

#apply a simple naive bayes on the text features
result=model.predict(tfidf.transform(orig_test2)) # these are ones we don't know the language

result=pd.DataFrame(result, index=orig_test)
result.columns=['language']
# can we establish the accuracy??
print("RESULT describe:", result.describe())
print("RESULT head:", result.head())
print("RESULT shape:", result.shape)
print("RESULT:", result)
print("RESULT fr]:", result[result['language']=='fr'])
print("RESULT es]:", result[result['language']=='es'])
print("RESULT ru]:", result[result['language']=='ru'])
print("RESULT de]:", result[result['language']=='de'])
print("RESULT ja]:", result[result['language']=='ja'])
print("RESULT zh]:", result[result['language']=='zh'])

print("Counter result language:", Counter(result.language))
# Lawrence: need to get this report and see how we can improve it...
print("CLASSIFICATION REPORT:")
print("OT:",len(orig_target),type(orig_target))
print("ROT:", result_on_training)
print(metrics.classification_report(orig_target,result_on_training))
# Lawrence: NEED TO SEE HOW ACCURATE it is and how to improve it....
# Lawrence: STOP HERE until we have a confident result





#result will be used later to replace 'ts' and 'er' values
#we need to remove train.language so that the train can be flattened with melt
del train['language']

melt_columns=49  # 49 gives best result compared with 48,50 and other combos
train = pd.melt(train[list(train.columns[-melt_columns:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
print("TH:", train.head())

train['date'] = train['date'].astype('datetime64[ns]')
train['weekend'] = ((train.date.dt.dayofweek) >=5).astype(float)
train['Monday'] = ((train.date.dt.dayofweek) ==0).astype(float)
train['Tuesday'] = ((train.date.dt.dayofweek) ==1).astype(float)
train['Wednesday'] = ((train.date.dt.dayofweek) ==2).astype(float)
train['Thursday'] = ((train.date.dt.dayofweek) ==3).astype(float)
train['Friday'] = ((train.date.dt.dayofweek) ==4).astype(float)
train['Saturday'] = ((train.date.dt.dayofweek) ==5).astype(float)
train['Sunday'] = ((train.date.dt.dayofweek) ==6).astype(float)
train['language']=train['Page'].apply(lambda x:re.split(".wikipedia.org", x)[0][-2:])
print("AFTER MELTING...")
print("train describe:", train.describe())
print("train head:", train.head())
print("train:", train)
print("Counter train language:", Counter(train.language))

#let's join with result to replace 'ts' and 'er'
join=train.loc[train.language.isin(["ts","er"]), ['Page']]
join['language']=0 #init
join.index=join["Page"]
join.language=result
train.loc[train.language.isin(["ts","er"]), ['language']]=join.language.values #replace
print("AFTER Joining...")
print("train describe:", train.describe())
print("train head:", train.head())
print("train:", train)
print("Counter train language:", Counter(train.language))


#Lawrence: can we improve/validate that these days are correct?
#official non working days by country (manual search with google)
#I made a lot of shortcuts considering that only Us and Uk used english idiom, 
#only Spain for spanish, only France for french, etc
train_us=['2015-07-04','2015-11-26','2015-12-25']+\
['2016-07-04','2016-11-24','2016-12-26']
test_us=[]
train_uk=['2015-12-25','2015-12-28'] +\
['2016-01-01','2016-03-28','2016-05-02','2016-05-30','2016-12-26','2016-12-27']
test_uk=['2017-01-01']
train_de=['2015-10-03', '2015-12-25', '2015-12-26']+\
['2016-01-01', '2016-03-25', '2016-03-26', '2016-03-27', '2016-01-01', '2016-05-05', '2016-05-15', '2016-05-16', '2016-10-03', '2016-12-25', '2016-12-26']
test_de=['2017-01-01']
train_fr=['2015-07-14', '2015-08-15', '2015-11-01', '2015-11-11', '2015-12-25']+\
['2016-01-01','2016-03-28', '2016-05-01', '2016-05-05', '2016-05-08', '2016-05-16', '2016-07-14', '2016-08-15', '2016-11-01','2016-11-11', '2016-12-25']
test_fr=['2017-01-01']
train_ru=['2015-11-04']+\
['2016-01-01', '2016-01-02', '2016-01-03', '2016-01-04', '2016-01-05', '2016-01-06', '2016-01-07', '2016-02-23', '2016-03-08', '2016-05-01', '2016-05-09', '2016-06-12', '2016-11-04']
test_ru=['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07', '2017-02-23']
train_es=['2015-08-15', '2015-10-12', '2015-11-01', '2015-12-06', '2015-12-08', '2015-12-25']+\
['2016-01-01', '2016-01-06', '2016-03-25', '2016-05-01', '2016-08-15', '2016-10-12', '2016-11-01', '2016-12-06', '2016-12-08', '2016-12-25']
test_es=['2017-01-01', '2017-01-06']
train_ja=['2015-07-20','2015-09-21', '2015-10-12', '2015-11-03', '2015-11-23', '2015-12-23']+\
['2016-01-01', '2016-01-11', '2016-02-11', '2016-03-20', '2016-04-29', '2016-05-03', '2016-05-04', '2016-05-05', '2016-07-18', '2016-08-11', '2016-09-22', '2016-10-10', '2016-11-03', '2016-11-23', '2016-12-23']
test_ja=['2017-01-01', '2017-01-09', '2017-02-11']
train_zh=['2015-09-27', '2015-10-01', '2015-10-02','2015-10-03','2015-10-04','2015-10-05','2015-10-06','2015-10-07']+\
['2016-01-01', '2016-01-02', '2016-01-03', '2016-02-08', '2016-02-09', '2016-02-10', '2016-02-11', '2016-02-12', '2016-04-04', '2016-05-01', '2016-05-02', '2016-06-09', '2016-06-10', '2016-09-15', '2016-09-16', '2016-10-03', '2016-10-04','2016-10-05','2016-10-06','2016-10-07']
test_zh=['2017-01-02', '2017-02-27', '2017-02-28', '2017-03-01']
#in China some saturday and sundays are worked
train_o_zh=['2015-10-10','2016-02-06', '2016-02-14', '2016-06-12', '2016-09-18', '2016-10-08', '2016-10-09']
test_o_zh=['2017-01-22', '2017-02-04']


train.loc[(train.language=='en')&(train.date.isin(train_us+train_uk)), 'holiday']=1
train.loc[(train.language=='de')&(train.date.isin(train_de)), 'holiday']=1
train.loc[(train.language=='fr')&(train.date.isin(train_fr)), 'holiday']=1
train.loc[(train.language=='ru')&(train.date.isin(train_ru)), 'holiday']=1
train.loc[(train.language=='es')&(train.date.isin(train_es)), 'holiday']=1
train.loc[(train.language=='ja')&(train.date.isin(train_ja)), 'holiday']=1
train.loc[(train.language=='zh')&(train.date.isin(train_zh)), 'holiday']=1
train.loc[(train.language=='zh')&(train.date.isin(train_o_zh)), 'holiday']=0
train.loc[train.holiday.isnull(), 'holiday'] = 0

#same with test
test = pd.read_csv("../input/key_1.csv")
test['date'] = test.Page.apply(lambda a: a[-10:])
test['Page'] = test.Page.apply(lambda a: a[:-11])
test['date'] = test['date'].astype('datetime64[ns]')
test['weekend'] = ((test.date.dt.dayofweek) >=5).astype(float)
test['Monday'] = ((test.date.dt.dayofweek) ==0).astype(float)
test['Tuesday'] = ((test.date.dt.dayofweek) ==1).astype(float)
test['Wednesday'] = ((test.date.dt.dayofweek) ==2).astype(float)
test['Thursday'] = ((test.date.dt.dayofweek) ==3).astype(float)
test['Friday'] = ((test.date.dt.dayofweek) ==4).astype(float)
test['Saturday'] = ((test.date.dt.dayofweek) ==5).astype(float)
test['Sunday'] = ((test.date.dt.dayofweek) ==6).astype(float)
test['language']=test['Page'].apply(lambda x:re.split(".wikipedia.org", x)[0][-2:])

#joint with result
join=test.loc[test.language.isin(["ts","er"]), ['Page']]
join['language']=0
join.index=join["Page"]
join.language=result
test.loc[test.language.isin(["ts","er"]), ['language']]=join.language.values

test.loc[(test.language=='en')&(test.date.isin(test_us+test_uk)), 'holiday']=1
test.loc[(test.language=='de')&(test.date.isin(test_de)), 'holiday']=1
test.loc[(test.language=='fr')&(test.date.isin(test_fr)), 'holiday']=1
test.loc[(test.language=='ru')&(test.date.isin(test_ru)), 'holiday']=1
test.loc[(test.language=='es')&(test.date.isin(test_es)), 'holiday']=1
test.loc[(test.language=='ja')&(test.date.isin(test_ja)), 'holiday']=1
test.loc[(test.language=='zh')&(test.date.isin(test_zh)), 'holiday']=1
test.loc[(test.language=='zh')&(test.date.isin(test_o_zh)), 'holiday']=0
test.loc[test.holiday.isnull(), 'holiday'] = 0

print("SAVING FILES:")
train.to_csv('train.csv')
test.to_csv('test.csv')
train.loc[train.Visits.isnull(), 'Visits'] = 0
train_last_entry=train[train.date=='2016-12-31'] #lets take max day
print("TRAIN RESULT:", train_last_entry)
print(train.head())
print(train)
train_page_per_dow = train.groupby(['Page','weekend','holiday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','language']).median().reset_index()
print("TPPDOW shape:", train_page_per_dow.shape)
print("TPPDOW:", train_page_per_dow.head())
print("TLE shape:", train_last_entry.shape())
print("TLE:", train_last_entry.head())
# call fast_smape on train_page_per_dow vs train_last_entry
#WE need to align the rows of train_page_per_dow and train_last_entry GP
res = smape_fast(train_page_per_dow['Visits'],train_last_entry['Visits']) # presuming the dates are aligned???
print("FAST SMAPE:", res) #GP need to see if SMAPE score is useful to predict the Kaggle public LB score, and alteast moves in the same direction and 1:1

test = test.merge(train_page_per_dow, how='left')

test.loc[test.Visits.isnull(), 'Visits'] = 0
test['Visits']=(test['Visits']*10).astype('int')/10
test[['Id','Visits']].to_csv('groupedBy.csv', index=False) ### current result gives SMAPE of 52.1, why did it get worse than before with more features? #GP
# GP flatten out the extra features before running the median operation on line 259, and see if you can recover the previous best score of 44.9


##### NOW TRY THIS APPROACH BUT REPLACE median with fibonnaci window function GP on line 259

######################## NOW RUN XGBOOST ###############################
### we need to added moving average time features here.... GP
sys.exit()
import xgboost
train.loc[:, 'Visits'] = train.Visits.fillna(0)

X_test = data.loc[data.eval_set == "test",:] # TBD pick a subset of the training to use for testing, say 10%
TBD remove the testing cases from train dataframe

# 
X_train = train.drop('Visits', axis=1)
y_train = train.Visits

d_train = xgboost.DMatrix(X_train, y_train)
xgb_params = {
    "objective"         : "reg:linear"   # TBD check ...
    ,"eval_metric"      : "fast_smape"   # TBD need custom smape get working in here, or as close as possible
    ,"eta"              : 0.09
    ,"max_depth"        : 7
    ,"min_child_weight" : 9
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}

watchlist= [(d_train, "train")]
model = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=700, evals=watchlist, verbose_eval=10)

predicted = model.predict(xgboost.DMatrix(test))
 
#now compare predicted to actual to get SMAPE

# now rerun model on full training data set so we can make Kaggle submission



