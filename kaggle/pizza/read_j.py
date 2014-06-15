import pandas as pd
loc_train = "data/1.json"
loc_test = "data/2.json"

df_train = pd.read_json(loc_train)
df_test = pd.read_json(loc_test)
print df_train
label_name = 'requester_received_pizza' # predict this
row_id_name = 'request_id' # row id, will be used in producing the prediction/output file

###############################################################

skip_columns = [label_name] + [row_id_name]
feature_cols = [col for col in df_train.columns if col not in skip_columns]
X_train = df_train[feature_cols]
X_test = df_test[feature_cols]


