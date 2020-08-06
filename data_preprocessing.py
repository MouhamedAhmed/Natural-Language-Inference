from data_loader import *
from word2vec import *
import torch
import json

# b2v = batch2vec('GoogleNews-vectors-negative300.bin',DEVICE)
w2v = word2vec('GoogleNews-vectors-negative300.bin')



def refine_data():
    out_data = []
    paths = ['snli_1.0_train.jsonl','snli_1.0_dev.jsonl']
    for path in paths:
        data = []
        for line in open('snli/'+path, 'r'):
            l = json.loads(line)

            sentence1 = split_sentence(l['sentence1'])
            sentence2 = split_sentence(l['sentence2'])

            sentence1 = delete_unk_words(sentence1,w2v)
            sentence2 = delete_unk_words(sentence2,w2v)

            if  len(sentence1) == 0 or len(sentence2) == 0:
                continue

            if l ['gold_label'] == 'entailment':
                label = 0
            elif l ['gold_label'] == 'neutral':
                label = 1
            elif l ['gold_label'] == 'contradiction':
                label = 2
            d = {
                'sentence1':l['sentence1'],
                'sentence2':l['sentence2'],
                'label':label,
                'sentence1_arr':sentence1,
                'sentence2_arr':sentence2
            }
            data.append(d)
        out_data.append(data)

    return out_data[0], out_data[1]




def split_sentence(sentence):
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
    sentence = sentence.translate(translator)
    sentence = sentence.strip()
    sentence = sentence.lower()
    sentence = sentence.replace('\n', '')
    words = sentence.split()
    return words

def delete_unk_words(sentence,w2v):
    i = 0
    length = len(sentence)
    while i < length:
        word_vec = w2v.get_embedding(sentence[i])
        if word_vec == None:
            del sentence[i]
            i -= 1
            length -= 1
        i += 1
    return sentence

def save_refined_data():
    train, test = refine_data()
    print(len(train))
    print(len(test))
    with open('train_refined.json', 'w') as fout:
        json.dump(train, fout)
    with open('dev_refined.json', 'w') as fout:
        json.dump(test, fout)

save_refined_data()

