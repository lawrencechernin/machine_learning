
dataPath = "data/"

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor



def loadData(datafile):
	return pd.read_csv(datafile)



def splitDatetime(data):
	sub = pd.DataFrame(data.datetime.str.split(' ').tolist(), columns = "date time".split())
	date = pd.DataFrame(sub.date.str.split('-').tolist(), columns="year month day".split())
	time = pd.DataFrame(sub.time.str.split(':').tolist(), columns = "hour minute second".split())
	data['year'] = date['year']
	data['month'] = date['month']
	data['day'] = date['day']
	data['hour'] = time['hour'].astype(int)
	return data



def createDecisionTree():
	est = DecisionTreeRegressor()
	return est

def createRandomForest():
	est = RandomForestRegressor(n_estimators=100)
	return est



def createExtraTree():
	est = ExtraTreesRegressor()
	return est



def predict(est, train, test, features, target):
	est.fit(train[features], train[target])
	with open("submission-extratrees.csv", 'wb') as f:
		f.write("datetime,count\n")
		for index, value in enumerate(list(est.predict(test[features]))):
			f.write("%s,%s\n" % (test['datetime'].loc[index], int(value)))

def main():
	train = loadData(dataPath + "train.csv")
	test = loadData(dataPath + "test.csv")
	train = splitDatetime(train)
	test = splitDatetime(test)
	target = 'count'
	features = [col for col in train.columns if col not in ['datetime', 'casual', 'registered', 'count']]
	est = createExtraTree()
	predict(est, train, test, features, target)

if __name__ == "__main__":
	main()


