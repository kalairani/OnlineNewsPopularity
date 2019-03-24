import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

csv_filename = "data/OnlineNewsPopularity.csv"

df = pd.read_csv(csv_filename)

popular = df.shares >= 1400
unpopular = df.shares < 1400

df.loc[popular, 'shares'] = 1
df.loc[unpopular, 'shares'] = 0

features = list(df.columns[2:60])

X = df[features]
y = df['shares']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4)

model = RandomForestClassifier(criterion="entropy", max_depth=None)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(metrics.accuracy_score(prediction, y_test))
