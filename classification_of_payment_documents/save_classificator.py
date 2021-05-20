import joblib
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import numpy as numpy

def save_clf(clf, name_file):
    joblib.dump(clf, name_file)
    return name_file

def load_clf(name_file):
    clf = joblib.load(name_file)
    return clf

def clf_from_file(clf, sentence):
    new_sentence = sentence.data.tolist()
    while(len(new_sentence) < 231):
        new_sentence.append(0)

    arr = numpy.array(new_sentence)
    # print("Arr: ", arr)

    new_sentence = sparse.csr_matrix(arr)
    # print("csr: ", new_sentence)
    #
    # print("len example: ", len(new_sentence))
    # print("Example!: ", new_sentence)
    predicted = clf.predict(new_sentence)


    print("!!!!!", predicted)