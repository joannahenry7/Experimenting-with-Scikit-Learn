import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)
# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def plot_wave():
    plt.plot(X, y, 'o')
    plt.ylim(-3, 3)
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.show()

def wave_k_neighbors():
    # instantiate the model and set the number of neigbors to consider to 3
    reg = KNeighborsRegressor(n_neighbors=3)
    # fit the model using the training data and training targets
    reg.fit(X_train, y_train)
    # make predictions on the test set
    print "Test set predictions:\n{}".format(reg.predict(X_test))
    # evaluate the model
    score = reg.score(X_test, y_test)
    print "Test set R^2: {:.2f}".format(score)

def compare_n_neighbors():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # create 1000 data points, evenly spaced between -3 and 3
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)
    for n_neighbors, ax in zip([1, 3, 9], axes):
        # make predictions using 1, 3, or 9 neighbors
        reg = KNeighborsRegressor(n_neighbors=n_neighbors)
        reg.fit(X_train, y_train)
        ax.plot(line, reg.predict(line))
        ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
        ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

        ax.set_title('{} neighbor(s)\ntrain score: {:.2f} test score: {:.2f}'.format(
            n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
        ax.set_xlabel('Feature')
        ax.set_ylabel('Target')
    axes[0].legend(['Model predictions', 'Training data/target',
                    'Test data/target'], loc='best')
    plt.show()

#plot_wave()
wave_k_neighbors()
compare_n_neighbors()
