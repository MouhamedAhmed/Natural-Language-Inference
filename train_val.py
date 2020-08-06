import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import json


###############
# train iteration
def train(batch_size, model, cross_entropy_loss_criterion, optimizer, data_loader, b2v, device):
    '''
    Function for the training step of the training loop
    '''
    data_loader.reset()

    model.train()
    cross_entropy_loss_epoch = 0
    i = 0
    while True:
         # get batch
        batch = data_loader.get_training_batch(batch_size)
        if batch == []:
            break
        batch_vec = b2v.convert_batch_to_vec(batch)
        y_true = batch_vec[-1]
        batch_vec = batch_vec[:-1]

        optimizer.zero_grad()

        # Forward pass
        y_hat = model(batch_vec[0],batch_vec[1],batch_vec[2],batch_vec[3]).to(device)

        # loss
        cross_entropy_loss = cross_entropy_loss_criterion(y_hat, y_true) 
        cross_entropy_loss.to(device)
        cross_entropy_loss_epoch += cross_entropy_loss.item()       

        # Backward pass
        cross_entropy_loss.backward()
        optimizer.step()
        i += 1
    
    return model, optimizer, cross_entropy_loss_epoch


# validate 
def validate(batch_size, model, cross_entropy_loss_criterion, data_loader, b2v, device):
    '''
    Function for the validation step of the training loop
    '''
    data_loader.reset()

    model.eval()
    cross_entropy_loss_epoch = 0

    while True:
         # get batch
        batch = data_loader.get_testing_batch(batch_size)
        if batch == []:
            break
        
        batch_vec = b2v.convert_batch_to_vec(batch)
        y_true = batch_vec[-1]
        batch_vec = batch_vec[:-1]

        # Forward pass
        y_hat = model(batch_vec[0],batch_vec[1],batch_vec[2],batch_vec[3]).to(device)

        # loss
        cross_entropy_loss = cross_entropy_loss_criterion(y_hat, y_true) 
        cross_entropy_loss_epoch += cross_entropy_loss.item()       
    
    return model, cross_entropy_loss_epoch


def training_loop(model, b2v, data_loader, cross_entropy_loss_criterion, batch_size, optimizer, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # log file
    log_file = open("guru99.txt","w+")

    
    # Train model
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss = train(batch_size, model, cross_entropy_loss_criterion, optimizer, data_loader, b2v, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(batch_size, model, cross_entropy_loss_criterion, data_loader, b2v, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_train_accuracy(model, batch_size, b2v, data_loader, device)
            valid_acc = get_valid_accuracy(model, batch_size, b2v, data_loader, device)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')
            
            log_file.write(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

        
        if epoch%3 == 0:
            torch.save(model.state_dict(), 'model')
            

    log_file.close() 

    plot_losses(train_losses, valid_losses)

    torch.save(model.state_dict(), 'model')

    return model, optimizer, train_losses, valid_losses


############
# helper functions
def get_train_accuracy(model, batch_size, b2v, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    data_loader.reset()

    with torch.no_grad():
        model.eval()
        while True:
            # get batch
            batch = data_loader.get_testing_batch(batch_size)
            if batch == []:
                break
            
            batch_vec = b2v.convert_batch_to_vec(batch)
            y_true = batch_vec[-1]
            batch_vec = batch_vec[:-1]

            # Forward pass
            y_prob = model(batch_vec[0],batch_vec[1],batch_vec[2],batch_vec[3]).to(device)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()


    return correct_pred.float() / n


def get_valid_accuracy(model, batch_size, b2v, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    data_loader.reset()


    with torch.no_grad():
        model.eval()
        while True:
            # get batch
            batch = data_loader.get_training_batch(batch_size)
            if batch == []:
                break
            
            batch_vec = b2v.convert_batch_to_vec(batch)
            y_true = batch_vec[-1]
            batch_vec = batch_vec[:-1]

            # Forward pass
            y_prob = model(batch_vec[0],batch_vec[1],batch_vec[2],batch_vec[3]).to(device)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n

def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    # change the plot style to default
    plt.style.use('default')
    