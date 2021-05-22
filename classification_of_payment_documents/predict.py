from scipy import sparse
import numpy as numpy
import json

def predict(clf, sentence):

    new_sentence = sentence.data.tolist()
    while(len(new_sentence) < 6883):
        new_sentence.append(0)
    arr = numpy.array(new_sentence)
    new_sentence = sparse.csr_matrix(arr)

    predicted_proba = clf.predict_proba(new_sentence)
    predicted = clf.predict(new_sentence)

    dictionary = dict(zip(clf.classes_, predicted_proba[0]))
    print(dictionary)
    predicted = str(predicted[0])
    print(predicted)
    print(type(predicted))

    with open('data/name_of_code.txt') as json_file:
        data = json.load(json_file)

    name = data.get(predicted)
    print(name)


    sorted_values = sorted(dictionary.values())  # Sort the values
    sort_dictionary = {}

    for i in sorted_values:
        for k in dictionary.keys():
            if dictionary[k] == i:
                sort_dictionary[k] = dictionary[k]
                break

    print(sort_dictionary)

    return predicted, name, sort_dictionary
