# http://www.kaggle.com/c/random-acts-of-pizza/data?sampleSubmission.csv.zip

import pandas as pd
from sklearn import ensemble
import math,sys,resource
label_name = 'requester_received_pizza' # predict this
row_id_name = 'request_id' # row id, will be used in producing the prediction/output file

###############################################################

do_not_process_columns = [row_id_name,'giver_username_if_known','request_text','request_text_edit_aware','requester_username','request_title','post_was_edited','requester_subreddits_at_request','requester_user_flair','unix_timestamp_of_request','unix_timestamp_of_request_utc']
skip_columns = [label_name] + do_not_process_columns


if __name__ == "__main__":
  loc_train = "data/train.json"
  loc_test = "data/test.json"
  loc_submission = "rf.submission.csv"

  df_train = pd.read_json(loc_train)
  df_test = pd.read_json(loc_test)

  feature_cols = [col for col in df_train.columns if col not in skip_columns]
  test_cols = [col for col in df_test.columns if col not in do_not_process_columns]

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
  test_cols = [col for col in df_test.columns if col not in do_not_process_columns]


  X_train = df_train[feature_cols]
  print "Length of X_train:"
  print len(X_train)

  X_test = df_test[test_cols]
  print "Length of X_test:"
  print len(X_test)

  y = df_train[label_name]
  test_ids = df_test[row_id_name]
  #print "Y:"
  #print y
  #print "Train:"
  #print X_train
  print "Test:"
  print X_test

  num_features = len(df_train.columns)
  
  n_estimators = int(math.sqrt(num_features))
  n_estimators = 50
  print "num_features: ", num_features,", n_estimators: ",n_estimators
  
  clf = ensemble.RandomForestClassifier(n_estimators = n_estimators, n_jobs = -1)
  print " Now fitting ..."

  clf.fit(X_train, y)
  print " Done fitting"
  with open(loc_submission, "wb") as outfile:
    outfile.write(row_id_name + "," + label_name + "\n")
    for e, val in enumerate(list(clf.predict(X_test))):
      outfile.write("%s,%s\n"%(test_ids[e],val))

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

