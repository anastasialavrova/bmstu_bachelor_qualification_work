import joblib

def save_clf(clf, name_file):
    joblib.dump(clf, "save_classifications/" + name_file)


def load_clf(name_file):
    clf = joblib.load("save_classifications/" + name_file)
    return clf