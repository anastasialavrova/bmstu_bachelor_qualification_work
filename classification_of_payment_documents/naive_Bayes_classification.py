from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import warnings


def NB(X, name):
    warnings.filterwarnings('ignore')
    X_train, X_test, y_train, y_test = train_test_split(X, name, test_size=0.2, random_state=1)
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    print("X_test: ", X_test[0])

    predicted = clf.predict(X_test[0])
    cnt = 0
    for i in range(len(predicted)):
        print(i,": ", predicted[i], " ", y_test[i], "\n")
        if (predicted[i] != y_test[i]):
            cnt += 1
    print (cnt)
    #print(metrics.classification_report(y_test, predicted))
    return clf