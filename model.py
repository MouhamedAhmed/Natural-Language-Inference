import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from batch2vec import *
# from data_loader import *

class avgPool(nn.Module):    
    def __init__(self):
        super(avgPool,self).__init__()
        
    def forward(self,x,x_original_length):
        x = torch.sum(x,1)
        x_original_length = x_original_length.unsqueeze(1)
        x = torch.div(x, x_original_length)
        return x


class batchDescriptor(nn.Module):    
    def __init__(self,embedding_dim, hidden_dim, n_layers):
        super(batchDescriptor,self).__init__()
        # self.lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.BiLSTM = nn.LSTM(input_size = embedding_dim,
                                    hidden_size = hidden_dim,
                                    num_layers = n_layers, 
                                    bidirectional = True)
        self.AvgPool =  avgPool()

    def forward(self,x,x_original_length):
        # pack = torch.nn.utils.rnn.pack_padded_sequence(x, x_original_length, batch_first=True, enforce_sorted=False)
        x,_ = self.BiLSTM(x)
        # unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(pack)
        x = self.AvgPool(x,x_original_length)
        return x


class PremisesHypothesesEmbedding(nn.Module):
    def __init__(self):
        super(PremisesHypothesesEmbedding, self).__init__()
 
    def forward(self, premises_descriptor, hypotheses_descriptor):
        mul = premises_descriptor * hypotheses_descriptor
        diff = premises_descriptor - hypotheses_descriptor
        out = torch.cat([premises_descriptor,mul,diff,hypotheses_descriptor],1)
        return out


class FullyConnectedNet(nn.Module):
    def __init__(self,hidden_dim):
        super(FullyConnectedNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=8*hidden_dim, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            
            nn.Linear(in_features=256, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=16),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),

            nn.Linear(in_features=16, out_features=3),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.net(x)


class model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers):
        super(model, self).__init__()
        self.descriptor = batchDescriptor(embedding_dim, hidden_dim, n_layers)
        self.merge = PremisesHypothesesEmbedding()
        self.fc_net = FullyConnectedNet(hidden_dim)

    def forward(self, premises, hypotheses, premises_original_lengths, hypotheses_original_lenghts):
        premises_descriptor = self.descriptor(premises, premises_original_lengths)
        hypotheses_descriptor = self.descriptor (hypotheses, hypotheses_original_lenghts)
        merged_embedding = self.merge(premises_descriptor,hypotheses_descriptor)
        out = self.fc_net(merged_embedding)
        return out


# m = model(300,100,2)

# b2v = batch2vec('GoogleNews-vectors-negative300.bin')
# d = dataLoader()
# batch = d.get_testing_batch(10)
# p,h,pl,hl = b2v.convert_batch_to_vec(batch)
# print(p.size())
# o = m(p,h,pl,hl)
# print(o.size())
# print(o)

