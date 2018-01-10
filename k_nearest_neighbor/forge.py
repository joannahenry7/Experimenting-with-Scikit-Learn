import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# generate dataset
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def plot_forge():
    # plot dataset
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(["Class 0", "Class 1"], loc=4)
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()

def forge_k_neighbors():
    # instantiate the class and set k to 3
    clf = KNeighborsClassifier(n_neighbors=3)
    # fit classifier using training data
    clf.fit(X_train, y_train)
    # make predictions for test data
    y_pred = clf.predict(X_test)
    print "Test set predictions: ", y_pred
    # evaluate how well model generalizes
    score = clf.score(X_test, y_test)
    print "Test set accuracy: {:.2f}".format(score)

def forge_decision_boundaries():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for n_neighbors, ax in zip([1, 3, 9], axes):
        # the fit method returns the object self, so we can instantiate
        # and fit in one line
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title('{} neighbor(s)'.format(n_neighbors))
        ax.set_xlabel('feature 0')
        ax.set_ylabel('feature 1')
    axes[0].legend(loc=3)
    plt.show()

#plot_forge()
forge_k_neighbors()
forge_decision_boundaries()
