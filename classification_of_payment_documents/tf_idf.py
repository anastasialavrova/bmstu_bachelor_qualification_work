from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd

# def tf_idf(filtered_sentence):
#     cv = CountVectorizer()
#     tfidf_transformer = TfidfTransformer()
#     word_count_vector = cv.fit_transform(filtered_sentence)
#     X = tfidf_transformer.fit_transform(word_count_vector)
#     tf_idf = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
#     return X

def tf_idf(filtered_sentence):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(filtered_sentence)
    tf_idf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    return X
