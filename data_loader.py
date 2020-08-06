import json
import torch
import torch.nn as nn
from gensim.models import Word2Vec
import string
import numpy as np
import random

class dataLoader():
    def __init__(self):
        self.training_set, self.testing_set = self.load_data()
        self.training_set_size = len(self.training_set)
        self.testing_set_size = len(self.testing_set)
        self.reset()

    def load_data(self):
        out_data = []
        paths = ['train_refined.json','dev_refined.json']
        for path in paths:
            data = []
            loaded_data = json.load(open(path, 'r'))
            for l in loaded_data:
                
                d = {
                    'sentence1':l['sentence1'],
                    'sentence2':l['sentence2'],
                    'sentence1_arr':l['sentence1_arr'],
                    'sentence2_arr':l['sentence2_arr'],
                    'label':l['label']
                }
                data.append(d)
            out_data.append(data)

        random.shuffle(out_data[0])
        random.shuffle(out_data[1])

        return np.array(out_data[0]), np.array(out_data[1])

    def reset(self):
        self.remaining_of_training = self.training_set.shape[0]
        self.last_untaken_training_element = 0
        self.remaining_of_testing = self.testing_set.shape[0]
        self.last_untaken_testing_element = 0
    
    def get_training_batch (self,batch_size):
        if self.remaining_of_training <= 0:
            return []
        batch_size = min(batch_size,self.remaining_of_training)
        
        # get random indices = batch_size at max
        indices = np.arange(self.last_untaken_training_element, self.last_untaken_training_element + batch_size)
        self.last_untaken_training_element += batch_size
        self.remaining_of_training -= batch_size

        # get batch
        batch = self.training_set[indices]

        return batch

    def get_testing_batch (self,batch_size):
        if self.remaining_of_testing <= 0:
            return []
        batch_size = min(batch_size,self.remaining_of_testing)
        
        # get random indices = batch_size at max
        indices = np.arange(self.last_untaken_testing_element, self.last_untaken_testing_element + batch_size)
        self.last_untaken_testing_element += batch_size
        self.remaining_of_testing -= batch_size

        # get batch
        batch = self.testing_set[indices]

        return batch
    
    # def split_sentence(self,sentence):
    #     translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
    #     sentence = sentence.translate(translator)
    #     sentence = sentence.strip()
    #     sentence = sentence.lower()
    #     sentence = sentence.replace('\n', '')
    #     words = sentence.split()
    #     return words

# d = dataLoader()
# x = d.get_testing_batch(5)
# print(x)