import nltk
import re
from work_with_file import *
from nltk.corpus import stopwords
nltk.download('stopwords')
import pymorphy2


def filter_with_masks(description):
    regular_mask_quotes = "('.*.*?')"
    regular_mask_another_quotes = '(".*.*?")'
    regular_mask_company_names = '(зао)? ?(оао)? ?(ооо)? ?(ао)? ?(пао)?".*?"'
    mask_dd_yyyy = "((0[1-9]|1[0-2])(\/|\.)\d{4})"
    mask_dd_mm_yyyy = "((0[1-9]|[12]\d|3[01]).(0[1-9]|1[0-2]).[12]\d{3})"
    mask_dd_mm_yy = "((0[1-9]|[12]\d|3[01]).(0[1-9]|1[0-2]).\d{2})"
    mask_initials = "(([а-я]\.)([а-я]\.))"
    mask_salary = '((з\/пл)(ат)?)'

    descriptions = []
    company_names = []

    for item in description:
        # приведение к нижнему регистру
        item = item.lower()

        # разделение слов пробелом заместо запятых и точки с запятой
        item = item.replace(",", " ")
        item = item.replace(";", " ")

        # удаление даты по маске
        item = re.sub(mask_dd_mm_yyyy, '', item)
        item = re.sub(mask_dd_mm_yy, '', item)
        item = re.sub(mask_dd_yyyy, '', item)

        # запись названий компаний и пр. (то, что находится между кавычек) для последующей фильтрации
        index_begin = str.find(item, '"')
        index_end = str.find(item, '"', index_begin + 1)
        if (index_begin != -1 and index_end != -1):
            company_name = item[(index_begin + 1):index_end]
            company_names.append(company_name)

        # удаление всех наваний в кавычках
        item = re.sub(regular_mask_quotes, '', item)
        item = re.sub(regular_mask_another_quotes, '', item)
        item = re.sub(regular_mask_company_names, '', item)

        # удаление инициалов
        item = re.sub(mask_initials, '', item)

        # замена всех сокращений от заработной платы на "з/п"
        item = re.sub(mask_salary, 'з/п', item)

        # удаление скобочек
        index_begin = str.find(item, "(")
        index_end = str.find(item, ")")
        if (index_begin != -1 and index_end != -1 and index_end > index_begin):
            item = item[0:index_begin] + item[(index_begin + 1):(index_end)] + item[(index_end + 1):-1]

        # замена точек на пробелы
        item = item.replace(".", " ")

        descriptions.append(item)

    return descriptions, company_names

def filter_with_numbers(word):
    for symbol in word:
        if symbol.isdigit():
            return False
    return True

def word_in_stop_words(word, company_names, people_names):
    stop_words = set(stopwords.words('russian'))
    dict = ["ЗАО", "ОАО", "ООО", "АО", "ПАО", "зао", "оао", "ооо", "ао", "пао", "г"]
    dict_symbols = ["№", "-", "%", "(%)", "//"]

    if not word in stop_words and word != "" and not word in dict \
            and not word in people_names and not word in dict_symbols \
            and not word in company_names and filter_with_numbers(word):
        return True
    else:
        return False

def word_in_reductions(word):
    reductions = [
        ["вып", "выплата"],
        ["выпл", "выплата"],
        ["дог", "договор"],
        ["пен", "пенсионный"],
        ["пенс", "пенсионный"],
        ["фин", "финансовый"],
        ["ср", "средство"],
        ["ср в", "средство"],
        ["ср вами", "средство"],
        ["инвест", "инвестирование"],
        ["единовр", "единовременный"],
        ["единов", "единовременный"],
        ["един", "единовременный"],
        ["нак", "накопительный"],
        ["накоп", "накопительный"],
        ["накопит", "накопительный"],
        ["страх", "страхование"],
        ["стр", "страхование"],
        ["сроч", "срочный"],
        ["инд", "индивидульный"],
        ["упл", "уплата"],
        ["взн", "взнос"],
        ["застр", "застрахованный"],
        ["уст", "установлен"],
        ["ден", "денежный"],
        ["опл", "оплата"],
        ["усл", "услуга"],
        ["оказ", "оказанный"],
        ["спец", "специальный"],
        ["вознаг", "вознаграждение"],
        ["управ", "управляющий"],
        ["управ ком", "управляющий компания"],
        ["комп", "компания"],
        ["обсл", "обслуживание"],
        ["согл", "согласно"],
        ["накопл", "накопление"],
        ["ком", "компенсация"],
        ["компенс", "компенсация"],
        ["перев", "перевод"],
        ["пер", "перевод"],
        ["уч", "учет"],
        ["провед", "проведение"],
        ["депоз", "депозит"],
        ["аукц", "аукцион"],
        ["рез", "резерв"],
        ["з пл", "з/п"],
        ["з платы", "з/п"],
        ["з п", "з/п"],
        ["зар", "з/п"],
        ["пл", "плата"],
        ["переч", "перечисление"],
        ["удерж е", "удержание"],
        ["удерж", "удержанный"],
        ["удержан", "удержанный"],
        ["р с", "расчетный счет"],
        ["сум", "сумма"],
        ["сч", "счет"],
        ["кл", "клиент"],
        ["обеспеч", "обеспечивающий"],
        ["осуществ", "осуществление"],
        ["оформ", "оформление"],
        ["страх акт", "страховой акт"],
        ["кол во", "количество"],
        ["цб", "ценный бумага"],
        ["распор", "распоряжение"],
        ["л счет", "личный счет"],
        ["л счета", "личный счет"],
        ["п п", "платежное поручение"],
        ["физ лиц", "физических лиц"],
        ["матер", "материальный"],
        ["предпринимат", "предпринимательский"],
        ["исполнит", "исполнитель"],
        ["кд", "корпоративный действие"],
        ["сро", "саморегулируемая организация"],
        ["брок", "брокерский"]
    ]

    for item in reductions:
        if (item[0] == word):
            word = item[1]

    return word

def filter_with_stop_words(descriptions, company_names, people_names):
    morph = pymorphy2.MorphAnalyzer()
    total = []
    for item in descriptions:
        filtered_sentence = []
        sentence = item.split(" ")
        for word in sentence:
            word = word_in_reductions(word)
            word = morph.parse(word)[0].normal_form
            if word_in_stop_words(word, company_names, people_names):
                filtered_sentence.append(word)
        total_sentence = ' '.join(filtered_sentence)
        total.append(total_sentence)
        #print(total_sentence)
    # print(total)
    return total


def filter(description):
    people_names = get_names()
    descriptions, company_names = filter_with_masks(description)
    result = filter_with_stop_words(descriptions, company_names, people_names)
    return result
