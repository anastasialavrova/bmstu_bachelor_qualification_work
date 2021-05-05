from sklearn.feature_selection import mutual_info_classif
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
    print("Filtered sentences: ", total)
    return total

def union_sentences(filtered_sentences):
    total = {}
    for item in filtered_sentences:
        doc = item.split(" ")
        total = set(total).union(set(doc))
    print("Total:", total)
    return total

def mutual_information(words):
    name_words = name[0].split(" ")
    res = mutual_info_classif(words, name_words, discrete_features='auto')
    print(res)

def MI(words):
    min_df = 5
    name_words = name[0].split(" ")
    print("Name: ", name_words)
    # remove english stop words (words that most likely do not have
    # anything to do with the document class because they occur everywhere, e.g. 'and')
    binary = True
    stop_words = None
    vectorizer = CountVectorizer(binary=binary, stop_words=stop_words, min_df=min_df)
    X_train = vectorizer.fit_transform(words)
    y_train = np.array(name_words).reshape((-1,))
    lexicon = vectorizer.get_feature_names()
    print("X: ", X_train)
    print("Y: ", y_train)
    res = mutual_info_classif(X_train, y_train, discrete_features='auto')
    print(res)

def calc_entropy(marginals : dict):
    """
    Given a dict of marginal probabilities calculate entropy
    :param marginals:
    :return:
    """
    h = 0
    for p in marginals:
        if marginals[p] == 0:
            continue
        h += marginals[p] * log2(marginals[p])
    h *= -1
    return h

def calc_conditional_entropy_over_all_x(df, px):
    h = 0
    gs = df.groupby('x').groups
    for x in gs:
        h += px[x] * calc_conditional_entropy_x(df, x)
    return h

def calc_conditional_entropy_x(df, x_value):
    vec_df = df[df['x'] == x_value].copy()
    vec_df['prob'] = vec_df['prob'] / vec_df['prob'].sum()

    h = 0
    for prob in vec_df['prob']:
        if prob != 0:
            h += prob * log2(prob)
    if h != 0:
        h *= -1
    return h

def calc_mutual_information_using_cond_entropy(df, px, py):
    """
    :param df: data frame
    :param px: dictionary of marginal probabilities for x
    :param py: dictionary of marginal probabilities for y
    :return: mutual information
    """
    hy = calc_entropy(py)
    h = calc_conditional_entropy_over_all_x(df, px)
    mi = hy - h
    return mi

def calc_mutual_information_for_word(df):
    # Calc marginal probabilities for x
    gs = df.groupby('x').groups
    px = {}
    for g in gs:
        px[g] = df.iloc[gs[g]].sum()['prob']

    # Calc marginal probabilities for y
    gs = df.groupby('y').groups
    py = {}
    for g in gs:
        py[g] = df.iloc[gs[g]].sum()['prob']

    hx = calc_entropy(px)
    hy = calc_entropy(py)

    mi = calc_mutual_information_using_cond_entropy(df, px, py)

    return mi


def main():
    filtered_sentence = filter()

    words = union_sentences(filtered_sentence)
    df = p.DataFrame(list(words))
    words = list(words)
    print("Total in array: ", words)

    # mi = MI(words)
    # print(mi)

    mi2 = calc_mutual_information_for_word(df)
    print(mi2)

if __name__ == "__main__":
    main()