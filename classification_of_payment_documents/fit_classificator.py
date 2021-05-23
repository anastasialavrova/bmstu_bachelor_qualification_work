from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
from sklearn.preprocessing import StandardScaler


def NB(X, name):
    warnings.filterwarnings('ignore')
    X_train, X_test, y_train, y_test = train_test_split(X, name, test_size=0.2, random_state=1)

    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(metrics.classification_report(y_test, predicted))
    return clf

def logistic_regression(X, name):
    warnings.filterwarnings('ignore')
    X_train, X_test, y_train, y_test = train_test_split(X, name, test_size=0.2, random_state=1)

    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    print(metrics.classification_report(y_test, predicted))
    return clf