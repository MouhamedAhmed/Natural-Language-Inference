import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_val import *
from model import *
from data_loader import *
from batch2vec import *

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 16
N_EPOCHS = 100

print('a7a main')
# instantiate the model
torch.manual_seed(RANDOM_SEED)
print('a7a main')

b2v = batch2vec('GoogleNews-vectors-negative300.bin',DEVICE)
print('a7a main')

data_loader = dataLoader()
print('a7a main')

embedding_dim = b2v.embedding_dim
hidden_dim = 100
n_layers = 2

model = model(embedding_dim, hidden_dim, n_layers).to(DEVICE)
print('a7a main')

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
cross_entropy_loss_criterion = nn.CrossEntropyLoss()

# start training
print('a7a main')

model, optimizer, train_losses, valid_losses = training_loop(model, b2v, data_loader, cross_entropy_loss_criterion, BATCH_SIZE, optimizer, N_EPOCHS, DEVICE)
 

