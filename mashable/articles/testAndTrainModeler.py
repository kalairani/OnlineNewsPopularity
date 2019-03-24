from sklearn import model_selection


def prepareTrainAndTestData(df,testsize):
    popular = df.shares >= 1400
    unpopular = df.shares < 1400

    df.loc[popular, 'shares'] = 1
    df.loc[unpopular, 'shares'] = 0

    features = list(df.columns[2:60])
    X = df[features]
    y = df['shares']
    return model_selection.train_test_split(X, y, test_size=testsize)

