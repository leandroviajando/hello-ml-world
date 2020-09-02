import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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


def cross_validate(models):
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
    cross_validate(MODELS)

    model = SVC(gamma="auto")
    print_results_stats(model)
