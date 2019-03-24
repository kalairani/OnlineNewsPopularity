from sklearn import metrics
import mashable.articles.readData as Reader
import mashable.articles.testAndTrainModeler as TestAndTrainModeler
import mashable.articles.randomForestTreeModel as randomForestTreeModel
import mashable.articles.decisionTreeModel as DecisionTreeModel
import mashable.articles.logisticRegressionTreeModel as LogisticModel
import mashable.articles.GaussianModel as GaussianModel


df = Reader.read("data/OnlineNewsPopularity.csv")
X_train, X_test, y_train, y_test = TestAndTrainModeler.prepareTrainAndTestData(df,0.4)
randomForestAccuracyScore = metrics.accuracy_score(randomForestTreeModel.predict(X_train,X_test,y_train), y_test)
decisionTreeAccuracyScore = metrics.accuracy_score(DecisionTreeModel.predict(X_train,X_test,y_train), y_test)
logisticRegressionAccuracyScore = metrics.accuracy_score(LogisticModel.predict(X_train,X_test,y_train), y_test)
gaussianAccuracyScore = metrics.accuracy_score(GaussianModel.predict(X_train,X_test,y_train), y_test)


print("randomForestAccuracyScore", randomForestAccuracyScore)
print("decisionTreeAccuracyScore", decisionTreeAccuracyScore)
print("logisticRegressionAccuracyScore", logisticRegressionAccuracyScore)
print("gaussianAccuracyScore", gaussianAccuracyScore)

if randomForestAccuracyScore > max(decisionTreeAccuracyScore, logisticRegressionAccuracyScore, gaussianAccuracyScore):
    print("\n\nRandom Forest Tree Model is better fit for prediction")
elif decisionTreeAccuracyScore > max(logisticRegressionAccuracyScore, gaussianAccuracyScore):
    print("Decision Tree Model looks better for prediction")
elif logisticRegressionAccuracyScore > gaussianAccuracyScore:
    print("Logistic Regression Model looks better for prediction")
else:
    print("Gaussian Model looks better for prediction")

