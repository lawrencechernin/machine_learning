#gets in the top 100 ranks.

import pandas as pd
from sklearn import ensemble
import math

if __name__ == "__main__":
  loc_train = "train.csv"
  loc_test = "test.csv"
  loc_submission = "kaggle.forest.submission.csv"

  df_train = pd.read_csv(loc_train)
  df_test = pd.read_csv(loc_test)

  feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

  X_train = df_train[feature_cols]
  X_test = df_test[feature_cols]
  print "X_train:"
#  print X_train
  y = df_train['Cover_Type']
  test_ids = df_test['Id']

  num_features = len(df_train.columns)

  n_estimators = int(math.sqrt(num_features))
  print "num_features: ", num_features,", n_estimators: ",n_estimators

  clf = ensemble.RandomForestClassifier(n_estimators = n_estimators, n_jobs = -1)

  clf.fit(X_train, y)

  with open(loc_submission, "wb") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(clf.predict(X_test))):
      outfile.write("%s,%s\n"%(test_ids[e],val))

