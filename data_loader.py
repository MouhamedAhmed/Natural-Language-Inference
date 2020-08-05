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
        self.reset()

    def load_data(self):
        out_data = []
        paths = ['snli_1.0_train.jsonl','snli_1.0_dev.jsonl']
        for path in paths:
            data = []
            for line in open('snli/'+path, 'r'):
                l = json.loads(line)
                if l ['gold_label'] == 'entailment':
                    label = [1,0,0]
                elif l ['gold_label'] == 'neutral':
                    label = [0,1,0]
                elif l ['gold_label'] == 'contradiction':
                    label = [0,0,1]
                d = {
                    'sentence1':l['sentence1'],
                    'sentence2':l['sentence2'],
                    'label':label
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

# d = dataLoader()
# x = d.get_testing_batch(5)
# print(x)