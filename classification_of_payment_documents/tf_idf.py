from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd

def tf_idf(filtered_sentence):
    cv = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    word_count_vector = cv.fit_transform(filtered_sentence)
    X = tfidf_transformer.fit_transform(word_count_vector)
    # feature_names = cv.get_feature_names()
    tf_idf = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
    print(tf_idf)
    return X