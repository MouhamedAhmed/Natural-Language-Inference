import string
from word2vec import *
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import numpy as np
from data_loader import *


class batch2vec():
    def __init__(self, path):
        self.w2v = word2vec(path)
    
    def convert_batch_to_vec(self,batch):
        premises = [e['sentence1'] for e in batch]
        hypotheses = [e['sentence2'] for e in batch]

        premises_vecs = [self.sentence2vec(sentence) for sentence in premises]
        hypotheses_vecs = [self.sentence2vec(sentence) for sentence in hypotheses]

        premises_vecs = pad_sequence(premises_vecs,1)
        hypotheses_vecs = pad_sequence(hypotheses_vecs,1)

        print(premises_vecs.size())
        print(hypotheses_vecs.size())

    def sentence2vec(self,sentence):
        sentence = self.split_sentence(sentence)
        length = len(sentence)
        i = 0
        while i < length:
            word_vec = self.w2v.get_embedding(sentence[i])
            if word_vec == None:
                del sentence[i]
                i -= 1
                length -= 1
            i += 1

        sentence_vec = self.w2v.get_embedding(sentence[0])
        for word in sentence[1:]:
            word_vec = self.w2v.get_embedding(word)
            sentence_vec = torch.cat((sentence_vec, word_vec))

        return sentence_vec

    def split_sentence(self,sentence):
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
        sentence = sentence.translate(translator)
        sentence = sentence.strip()
        sentence = sentence.lower()
        sentence = sentence.replace('\n', '')
        words = sentence.split()
        return words

# b2v = batch2vec('GoogleNews-vectors-negative300.bin')
# d = dataLoader()
# batch = d.get_testing_batch(10)
# b2v.convert_batch_to_vec(batch)







