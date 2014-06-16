# http://www.kaggle.com/c/random-acts-of-pizza/data?sampleSubmission.csv.zip

import pandas as pd
from sklearn import ensemble
import math,sys,resource
import utils
label_name = 'requester_received_pizza' # predict this
row_id_name = 'request_id' # row id, will be used in producing the prediction/output file

###############################################################

do_not_process_columns = [row_id_name,'giver_username_if_known','request_text','request_text_edit_aware','requester_username','request_title','post_was_edited','requester_subreddits_at_request','requester_user_flair','unix_timestamp_of_request']
skip_columns = [label_name] + do_not_process_columns

# found by odds_ratio
top_text_words = ['rice','person','edit','paycheck','went','request','wife','currently','start','offer','check','recently'] # in receiving pizza
top_title_words = ['cooking','anniversary','asking','ask','christmas','happy','baby','father','arizona','daughter','kentucky','yesterday','laid','late'] # in title of receiving pizza



if __name__ == "__main__":
  loc_train = "data/train.json"
  loc_test = "data/test.json"
  #loc_train = "data/train_3.json"
  #loc_test = "data/test_3.json"
  loc_submission = "et.submission.csv"

  df_train = pd.read_json(loc_train)
  df_test = pd.read_json(loc_test)


  feature_cols = [col for col in df_train.columns if col not in skip_columns]
  test_cols = [col for col in df_test.columns if col not in do_not_process_columns]
  df_train = utils.feature_engineer_text(df_train,top_text_words,'request_text_edit_aware')  # do some feature engineering with request_text_edit_aware
  df_train = utils.feature_engineer_text(df_train,top_title_words,'request_title')  # do some feature engineering with request_title. appends new feature
  df_train = utils.feature_engineer_date(df_train) # put time of day,week,month,year into features
  print "DFTC1 before FE:", df_test.columns

  df_test = utils.feature_engineer_text(df_test,top_text_words,'request_text_edit_aware')  # do some feature engineering with request_text_edit_aware
  df_test = utils.feature_engineer_text(df_test,top_title_words,'request_title')  # do some feature engineering with request_title
  df_test = utils.feature_engineer_date(df_test) # put time of day,week,month,year into features
  print "DFTC2 after FE:", df_test.columns

  print "Checking if columns match..."
  print "feature_cols:", len(feature_cols), feature_cols
  print "----------------------------------------------------------"
  print "test_cols:", len(test_cols), test_cols
  for feature in feature_cols:
	if feature not in test_cols:
		print "oops, feature missing from test_cols : ", feature
		do_not_process_columns.append(feature)



  for feature in test_cols:
	if feature not in feature_cols:
		print "oops, feature missing from feature_cols : ", feature
		do_not_process_columns.append(feature)


  print "Revising columns..."
  skip_columns = [label_name] + do_not_process_columns
  feature_cols = [col for col in df_train.columns if col not in skip_columns]
  print "DFT3", df_test.columns
  test_cols = [col for col in df_test.columns if col not in do_not_process_columns]


  X_train = df_train[feature_cols]
  print "X_train shape: ", X_train.shape

  X_test = df_test[test_cols]
  print "X_test shape: ", X_test.shape

  y = df_train[label_name]
  test_ids = df_test[row_id_name]
  print df_test.columns
  #print "TEST IDS:", test_ids
  print "Shape of test_ids: ", test_ids.shape
  #print "Y:"
  #print y
  #print "Train:"
  #print X_train
  #print "Test:"
  #print X_test

  num_features = len(df_train.columns)
  
  n_estimators = int(math.sqrt(num_features))
  n_estimators = 8
  print "num_features: ", num_features,", n_estimators: ",n_estimators
  
  clf = ensemble.ExtraTreesClassifier(n_estimators = n_estimators, n_jobs = -1)
  print " Now fitting ..."

  clf.fit(X_train, y)

  print " Done fitting!"

  print "Feature Importances:" 
  index_of_feature = 0  # need to view by index
  for feature in feature_cols:
        importance = round(clf.feature_importances_[index_of_feature],4)
        print "Feature: ", feature, ", Importance: ", importance
	index_of_feature += 1

  with open(loc_submission, "wb") as outfile:
    outfile.write(row_id_name + "," + label_name + "\n")
    for e, val in enumerate(list(clf.predict(X_test))):
      if val :
	val_0_1 = 1
      else :
        val_0_1 = 0
      outfile.write("%s,%s\n"%(test_ids[e],val_0_1))

  print "Done!"

  # what is the usage:
  sys.exit()
  usage = resource.getrusage(resource.RUSAGE_SELF)

  for name, desc in [
    ('ru_utime', 'User time'),
    ('ru_stime', 'System time'),
    ('ru_maxrss', 'Max. Resident Set Size'),
    ('ru_ixrss', 'Shared Memory Size'),
    ('ru_idrss', 'Unshared Memory Size'),
    ('ru_isrss', 'Stack Size'),
    ('ru_inblock', 'Block inputs'),
    ('ru_oublock', 'Block outputs'),
    ]:
    print '%-25s (%-10s) = %s' % (desc, name, getattr(usage, name))

