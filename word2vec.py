import torch
import torchwordemb
import numpy as np
class word2vec:
    def __init__(self,path):
        self.vocab, self.embedding_matrix = torchwordemb.load_word2vec_bin(path)
    
    def get_embedding(self,word):
        if word in self.vocab:
            return self.embedding_matrix[self.vocab[word]].unsqueeze(0)
        return None


# w = word2vec('GoogleNews-vectors-negative300.bin')
# x = w.get_embedding('apple')
# print(x)


