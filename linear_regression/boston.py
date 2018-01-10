import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def boston_linear_regression():
    lr = LinearRegression().fit(X_train, y_train)
    print "Training set score: {:.2f}".format(lr.score(X_train, y_train))
    print "Test set score: {:.2f}".format(lr.score(X_test, y_test))

boston_linear_regression()
