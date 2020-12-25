import matplotlib.pyplot as plt
import pandas as pd
import pretty_errors  # noqa: F401
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
NAMES = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]

MODELS = [
    ("LR", LogisticRegression(solver="liblinear", multi_class="ovr")),
    ("LDA", LinearDiscriminantAnalysis()),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier()),
    ("NB", GaussianNB()),
    ("SVM", SVC(gamma="auto")),
]


def print_summary_stats(dataset):
    print(dataset.shape)
    print(dataset.head(20))
    print(dataset.describe())
    print(dataset.groupby("class").size())


def plot_summary_stats(dataset):
    dataset.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.show()

    dataset.hist()
    plt.show()

    scatter_matrix(dataset)
    plt.show()


def split_train_validation_sets(dataset):
    array = dataset.values
    X = array[:, 0:4]
    y = array[:, 4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, y, test_size=0.20, random_state=1
    )
    return X_train, X_validation, Y_train, Y_validation


def cross_validate(models, X_train, Y_train):
    results = []
    names = []

    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring="accuracy"
        )
        results.append(cv_results)
        names.append(name)
        print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

    plt.boxplot(results, labels=names)
    plt.title("Algorithm Comparison")
    plt.show()


def print_results_stats(model):
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


if __name__ == "__main__":
    dataset = pd.read_csv(URL, names=NAMES)
    print_summary_stats(dataset)
    plot_summary_stats(dataset)

    X_train, X_validation, Y_train, Y_validation = split_train_validation_sets(dataset)
    cross_validate(MODELS, X_train, Y_train)

    model = SVC(gamma="auto")
    print_results_stats(model)
