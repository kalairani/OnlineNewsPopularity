from sklearn.linear_model import LogisticRegression


def predict(x_train, x_test, y_train):
    model = LogisticRegression(dual=True)
    model.fit(x_train, y_train)
    return model.predict(x_test)

