from naive_Bayes_classification import NB
from selection_of_terms import *
from logistic_regression import *
from tf_idf import *
from work_with_file import *
from save_classificator import *


if __name__ == '__main__':
    name, description = read_data()
    filtered_sentence = filter(description)
    res = tf_idf(filtered_sentence)
    clf_NB = NB(res, name)
    # clf_lr = logistic_regression(res, name)

    name_file = save_clf(clf_NB, 'clf_NB.pkl')
    clf = load_clf(name_file)
    filtered_example = filter(['Программа для ЭВМ "Корпоративная система электронного документооборота DIRECTUM", в рамках технического решения "Обработка обращений клиентов" , дог. N 003/19 от 05.09.19г, счет N 0275/У от 27.04.20г.'])
    res_example = tf_idf(filtered_example)
    clf_from_file(clf, res_example)


