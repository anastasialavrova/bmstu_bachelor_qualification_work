from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from work_with_file import name, description
import re
from nltk.corpus import stopwords
import nltk
import pandas as p
import numpy as np
from math import log2

nltk.download('stopwords')
import matplotlib.pyplot as plt
import numpy as np
import csv
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest


# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None, delimiter='|')
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:, -1]
    # format all fields as string
    X = X.astype(str)
    return X, y


def prepare_inputs(X_train):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    return X_train_enc


# prepare target
def prepare_targets(y_train):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    return y_train_enc


def filter():
    regular_mask = "^(?:(?:31(\/|-|\.)(?:0?[13578]|1[02]))\1" \
                   "|(?:(?:29|30)(\/|-|\.)(?:0?[13-9]|1[0-2])\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.)0?2\3" \
                   "(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$" \
                   "|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.)(?:(?:0?[1-9])|(?:1[0-2]))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$"

    stop_words = set(stopwords.words('russian'))
    total = []
    for item in description:
        filtered_sentence = []
        sentence = item.split(" ")
        for word in sentence:
            word = word.lower()
            re.sub(regular_mask, '', word)
            if not word in stop_words and word != "":
                filtered_sentence.append(word)
        total_sentence = ' '.join(filtered_sentence)
        total.append(total_sentence)
    print(total)
    return total


def create_array(filetred_sentences):
    X_train = []
    for item in filetred_sentences:
        doc = item.split(" ")
        X_train.append(doc)
    X_train = np.array([np.array(xi) for xi in X_train])
    print("X_train: ", X_train)
    y_train = ["Получение денежных средств на р/с"] * len(X_train)
    print("y_train: ", y_train)
    return X_train, y_train


def mutual_information(X_train, y_train):
    mi = mutual_info_classif(X_train, y_train)
    print("MI: ", mi)
    return mi

def chi(X_train, y_train):
    chi = chi2(X_train, y_train)
    return chi


def create_csv(filtered_sentence):
    X_train = []
    for item in filtered_sentence:
        doc = item.split(" ")
        X_train.append(doc)
    y_train = ["Получение денежных средств на р/с"] * len(X_train)
    with open('filename.csv', 'w') as myfile:
        for item in X_train:
            myfile.write("'")
            myfile.write(' '.join([i for i in item]))
            myfile.write("'")
            myfile.write('|"' + str(y_train[0]) + '"\n')


def get_data():
    X, y = load_dataset('filename.csv')
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    print("X_train: ", X_train)
    print("y_train: ", y_train)
    X_train_enc = prepare_inputs(X_train)
    # prepare output data
    y_train_enc = prepare_targets(y_train)
    return X_train_enc, y_train_enc


def select_features(X_train, y_train):
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    return X_train_fs, fs


def main():
    filtered_sentence = filter()
    create_csv(filtered_sentence)
    X_train, y_train = get_data()
    mi = mutual_information(X_train, y_train)
    X_train_fs, fs = select_features(X_train, y_train)
    # what are scores for the features
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))

    chi_res = chi(X_train, y_train)
    print("CHI: ", chi_res)


if __name__ == "__main__":
    main()
