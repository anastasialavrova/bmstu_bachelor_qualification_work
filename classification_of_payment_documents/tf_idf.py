from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
import re
from work_with_file import name, description
from nltk.corpus import stopwords
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
    print(total)
    return total

def tf_function(filtered_sentence):
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(filtered_sentence)

    tf = pd.DataFrame(word_count_vector.toarray(), columns=cv.get_feature_names())
    return tf

def idf_function(filtered_sentence):
    tfidf_transformer = TfidfTransformer()
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(filtered_sentence)
    X = tfidf_transformer.fit_transform(word_count_vector)
    idf = pd.DataFrame({'feature_name': cv.get_feature_names(), 'idf_weights': tfidf_transformer.idf_})
    return idf

def tf_idf(filtered_sentence):
    cv = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    word_count_vector = cv.fit_transform(filtered_sentence)
    X = tfidf_transformer.fit_transform(word_count_vector)
    tf_idf = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
    print(tf_idf)
    return tf_idf

def main():
    filtered_sentence = filter()
    result = tf_idf(filtered_sentence)
    print(result)

if __name__ == "__main__":
    main()