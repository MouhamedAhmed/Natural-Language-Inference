import string
from word2vec import *
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import numpy as np
from data_loader import *
import copy


class batch2vec():
    def __init__(self, path, device):
        self.w2v = word2vec(path)
        self.embedding_dim = self.w2v.embedding_matrix.size()[1]
        self.device = device
    
    def convert_batch_to_vec(self,batch):
        premises = [e['sentence1_arr'] for e in batch]
        hypotheses = [e['sentence2_arr'] for e in batch]
        labels = [e['label'] for e in batch]

        premises_vecs = [self.sentence2vec(sentence) for sentence in premises]
        hypotheses_vecs = [self.sentence2vec(sentence) for sentence in hypotheses]

        premises_original_length = torch.tensor([p.size()[0] for p in premises_vecs]).to(self.device)
        hypotheses_original_length = torch.tensor([p.size()[0] for p in hypotheses_vecs]).to(self.device)
        labels = torch.tensor(labels).to(self.device)

        premises_vecs = pad_sequence(premises_vecs,1).to(self.device)
        hypotheses_vecs = pad_sequence(hypotheses_vecs,1).to(self.device)

        return premises_vecs, hypotheses_vecs, premises_original_length, hypotheses_original_length, labels
        
    def sentence2vec(self,sentence):
        length = len(sentence)
        
        # if(len(sentence) == 0):
        #     return []
        sentence_vec = self.w2v.get_embedding(sentence[0])
        
        if(len(sentence) == 1):
            return sentence_vec

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







