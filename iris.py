import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# load iris data and split into test set and train set
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                    iris_dataset['target'], random_state=0)

# create a scatter matrix from iris_dataset
def iris_scatter_matrix():
    # create dataframe from data in X_train
    # label the columns using the strings in iris_dataset.feature_names
    iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    # create a scatter matrix from the dataframe, color by y_train
    grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
            marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
    plt.show()

def iris_k_neighbors():
    # instantiate KNeighborsClassifier and set number of neighbors to 1
    knn = KNeighborsClassifier(n_neighbors=1)
    # build model on training set
    knn.fit(X_train, y_train)

    # evaluate the model
    score = knn.score(X_test, y_test)
    # alternate method:
    #y_pred = knn.predict(X_test)
    #score = np.mean(y_pred == y_test)
    print "Test set score: ", score

    # make predictions for data with unknown label
    X_new = np.array([[5, 2.9, 1, 0.2]])
    prediction = knn.predict(X_new)
    pred_target = iris_dataset['target_names'][prediction]
    print "Prediction: ", prediction
    print "Predicted target name: ", pred_target

#iris_scatter_matrix()
iris_k_neighbors()
