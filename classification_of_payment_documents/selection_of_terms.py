import string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
import re
from work_with_file import name, description
from nltk.corpus import stopwords
from russian_names import RussianNames
import matplotlib.pyplot as plt
import pandas as pd
nltk.download('stopwords')
import json
import pymorphy2

def filter(people_names):
    regular_mask_quotes = "'.*?'"
    regular_mask_another_quotes = '".*?"'
    regular_mask_company_names = '(ЗАО)? ?(ОАО)? ?(ООО)? ?(АО)? ?(ПАО)?".*?"'
    mask_dd_yyyy = "((0[1-9]|1[0-2])(\/|\.)\d{4})"
    mask_dd_mm_yyyy = "((0[1-9]|[12]\d|3[01]).(0[1-9]|1[0-2]).[12]\d{3})"
    mask_dd_mm_yy = "((0[1-9]|[12]\d|3[01]).(0[1-9]|1[0-2]).\d{2})"
    mask_lf = "(№? ?ЛФ-?\d{4}-\d{1,20})"
    mask_num = "(№? ?ДУ-П?\d{1,20})"
    mask_dp = "((№?)( ?)((-?)(ДП)?( ?))\d{1,10}(-?)\d{1,10}(-?)\d{1,10}(\/?)(\d{1,10})(-?)(\d{1,10})(-?)(\d{1,10})?)"
    mask_tp_du = "((ТП)?-?(ДУ)?-([А-Я]|[а-я])?\d{1,10}(\/)?(\d{1,10})?)"
    dict = ["ЗАО", "ОАО", "ООО", "АО", "ПАО", "зао", "оао", "ооо", "ао", "пао"]
    mask_numbers_with_slash = "(\d{1,20}/\d{1,20})"
    mask_numbers = "(\d{1,20})"
    dict_symbols = ["№", "-", "%", "(%)", "//"]
    mask_brackets = "(\(.*?\))"

    stop_words = set(stopwords.words('russian'))
    total = []
    descriptions = []
    for item in description:
        item = item.replace(",", " ")
        item = item.replace(";", " ")
        item = re.sub(mask_dd_mm_yyyy, '', item)
        item = re.sub(mask_dd_mm_yy, '', item)
        item = re.sub(mask_dd_yyyy, '', item)
        item = re.sub(regular_mask_quotes, '', item)
        item = re.sub(regular_mask_another_quotes, '', item)
        item = re.sub(regular_mask_company_names, '', item)
        item = re.sub(mask_lf, '', item)
        item = re.sub(mask_dp, '', item)
        item = re.sub(mask_tp_du, '', item)
        item = re.sub(mask_num, '', item)
        item = re.sub(mask_numbers_with_slash, '', item)
        item = re.sub(mask_numbers, '', item)
        # item = re.sub(mask_brackets, item[1:-1], item)
        index_begin = str.find(item, "(")
        index_end = str.find(item, ")")
        item = item[0:index_begin] + item[(index_begin + 1):(index_end)] + item[(index_end + 1):-1]
        item = item.replace(".", " ")
        descriptions.append(item)

    morph = pymorphy2.MorphAnalyzer()

    for item in descriptions:
        filtered_sentence = []
        sentence = item.split(" ")
        for word in sentence:
            word = word.lower()
            # word = morph.parse(word)[0].normal_form
            if not word in stop_words and word != "" and not word in dict and not word in people_names and not word in dict_symbols:
                filtered_sentence.append(word)
        total_sentence = ' '.join(filtered_sentence)
        total.append(total_sentence)
        print(total_sentence)
    print(total)
    return total

def tf_function(filtered_sentence):
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(filtered_sentence)
    tf = pd.DataFrame(word_count_vector.toarray(), columns=cv.get_feature_names())
    return tf

def tf_idf(filtered_sentence):
    cv = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    word_count_vector = cv.fit_transform(filtered_sentence)
    X = tfidf_transformer.fit_transform(word_count_vector)
    tf_idf = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
    print(tf_idf)
    return X

def create_set(description):
    total = {}
    for item in description:
        doc = item.split(" ")
        total = set(total).union(set(doc))
    total = list(total)
    print("Total:", total)

def get_names():
    f = open("names/female_names_rus.txt", "r")
    f2 = open("names/male_names_rus.txt", "r")
    f3 = open("names/male_surnames_rus.txt", "r")
    f4 = open("names/names3.txt", "r")
    female_names = []
    male_names = []
    male_surnames = []
    names = []
    for line in f:
        female_names.append(line[:-1].lower())
    for line in f2:
        male_names.append(line[:-1].lower())
    for line in f3:
        male_surnames.append(line[:-1].lower())
    for line in f4:
        names.append(line[:-1].lower())
    f.close()
    res = female_names + male_names + male_surnames + names
    # print(res)
    return res




def main():
    people_names = get_names()
    filtered_sentence = filter(people_names)
    set = create_set(filtered_sentence)
    r = tf_function(filtered_sentence)
    res = tf_idf(filtered_sentence)



if __name__ == "__main__":
    main()