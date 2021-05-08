import nltk

from tf_idf import res_X
from sklearn.naive_bayes import MultinomialNB
from work_with_file import name
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
nltk.download('stopwords')

def create_target():
    target = 1
    target_names = []
    target_train = []
    for i in range(len(name)):
        if i != 0:
            if name[i - 1] != name[i]:
                target = target + 1
                target_names.append(name[i])
            target_train.append(target)
        else:
            target_names.append(name[i])
            target_train.append(target)
    return target_train, target_names

def main():
    target_train, target_names = create_target()
    print(res_X)
    clf = MultinomialNB().fit(res_X, target_train)
    print(clf)
    stop_words = set(stopwords.words('russian'))
    docs_new = ['Оплата страхового взноса по Договору страхования № 0122130464484 от 05.09.2018, ФИО страхователя Ермошкина Екатерина Ивановна, сумма цифрами 550,00. НДС не облагается', 'Перечисление денежных средств по договору обслуживания на финансовых рынках № 1001233 от 10.04.2015г. для совершения операций с ценными бумагами на Московской бирже (субсчет']
    count_vect = CountVectorizer(stop_words=stop_words)
    X_new_counts = count_vect.transform(docs_new)
    tfidf_transformer = TfidfTransformer()
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, target_names[category]))

if __name__ == "__main__":
    main()




