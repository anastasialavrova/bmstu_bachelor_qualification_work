import pandas as pd
import math
import nltk
from work_with_file import name, description
from nltk.corpus import stopwords
nltk.download('stopwords')


def computeTF(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count/float(corpusCount)
    return(tfDict)


def computeIDF(docList):
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / (float(val) + 1))

    return (idfDict)

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return(tfidf)

total = {}
for item in description:
    doc = item.split(" ")
    total = set(total).union(set(doc))
print("Total:", total)

stop_words = set(stopwords.words('russian'))
filtered_sentence = [w for w in total if not w in stop_words] #(w and w.swapcase()) ??????
print("Filtered total:", filtered_sentence)

dictionaries = []
tf_array = []
for item in description:
    wordDict = dict.fromkeys(filtered_sentence, 0)
    for word in item:
        try:
            wordDict[word] += 1
        except:
            continue
    dictionaries.append(wordDict)
    tf = computeTF(wordDict, item)
    tf_array.append(tf)
# print("Dictionaries", dictionaries)

idfs = computeIDF([dict for dict in dictionaries])

idf_array = []
for tf in tf_array:
    idf = computeTFIDF(tf, idfs)
    idf_array.append(idf)

idf= pd.DataFrame([idf for idf in idf_array])
print(idf)



print("\n--------------------------------------------------------------------------------------------------\n")
#-------------------------------------------------


first_sentence = "Data Science is the sexiest job of the 21st century"
second_sentence = "machine learning is the key for data science"

#split so each word have their own string
first_sentence = first_sentence.split(" ")
second_sentence = second_sentence.split(" ") #join them to remove common duplicate words
total= set(first_sentence).union(set(second_sentence))
print(total)

wordDictA = dict.fromkeys(total, 0)
wordDictB = dict.fromkeys(total, 0)
for word in first_sentence:
    wordDictA[word] += 1

for word in second_sentence:
    wordDictB[word] += 1

pd.DataFrame([wordDictA, wordDictB])

# фильтрация слов-связок
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('russian'))
filtered_sentence = [w for w in wordDictA if not w in stop_words]
print(filtered_sentence)


tfFirst = computeTF(wordDictA, first_sentence)
tfSecond = computeTF(wordDictB, second_sentence)
tf = pd.DataFrame([tfFirst, tfSecond])
print(tf)

#inputing our sentences in the log file
idfs = computeIDF([wordDictA, wordDictB])

idfFirst = computeTFIDF(tfFirst, idfs)
idfSecond = computeTFIDF(tfSecond, idfs)
idf= pd.DataFrame([idfFirst, idfSecond])
print(idf)
