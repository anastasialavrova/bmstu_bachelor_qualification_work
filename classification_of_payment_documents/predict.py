from scipy import sparse
import numpy as numpy
import json
from sklearn.preprocessing import StandardScaler

def predict(clf, sentence):
    scaler = StandardScaler(with_mean=False)
    scaler.fit(sentence)
    sentence = scaler.transform(sentence)

    new_sentence = sentence.data.tolist()
    while(len(new_sentence) < 6875): #6883 #231 #6875
        new_sentence.append(0)
    arr = numpy.array(new_sentence)
    new_sentence = sparse.csr_matrix(arr)

    predicted_proba = clf.predict_proba(new_sentence)
    predicted = clf.predict(new_sentence)

    dictionary = dict(zip(clf.classes_, predicted_proba[0]))
    predicted = str(predicted[0])

    with open('data/name_of_code.txt') as json_file:
        data = json.load(json_file)

    name = data.get(predicted)


    sorted_values = reversed(sorted(dictionary.values()))
    sort_dictionary = {}

    for i in sorted_values:
        for k in dictionary.keys():
            if dictionary[k] == i:
                sort_dictionary[k] = dictionary[k]
                break

    res = sort_dictionary

    return predicted, name, res
