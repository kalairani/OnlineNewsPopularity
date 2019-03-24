from sklearn.naive_bayes import GaussianNB


def predict(x_train, x_test, y_train):
    model = GaussianNB()
    model.fit(x_train, y_train)
    return model.predict(x_test)
