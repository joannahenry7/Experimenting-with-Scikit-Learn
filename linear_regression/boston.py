import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def boston_linear_regression():
    lr = LinearRegression().fit(X_train, y_train)
    print "Training set score: {:.2f}".format(lr.score(X_train, y_train))
    print "Test set score: {:.2f}".format(lr.score(X_test, y_test))

def boston_ridge_regression():
    ridge = Ridge().fit(X_train, y_train)
    print "Training set score: {:.2f}".format(ridge.score(X_train, y_train))
    print "Test set score: {:.2f}".format(ridge.score(X_test, y_test))

    # alpha parameter specifies how much importance the model places on simplicity
    # versus training set performance; default is 1.0
    # Increasing alpha forces coefficients to move closer to zero
    ridge10 = Ridge(alpha=10).fit(X_train, y_train)
    print "Training set score: {:.2f}".format(ridge10.score(X_train, y_train))
    print "Test set score: {:.2f}".format(ridge10.score(X_test, y_test))

    ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
    print "Training set score: {:.2f}".format(ridge01.score(X_train, y_train))
    print "Test set score: {:.2f}".format(ridge01.score(X_test, y_test))

boston_linear_regression()
boston_ridge_regression()
