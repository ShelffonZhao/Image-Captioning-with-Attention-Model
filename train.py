import os
import random

import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

from data_loader import data_loader
from model import caption_lstm
from utils import *

# Hyper-parameters
BATCH_SIZE = 30
INPUT_CHANNEL = 512 # input dimensions
LEARNING_RATE = 1e-3
NUM_EPOCH = 20
DROP_OUT = 0.5
TOTAL_PIC = 10000  # number of pictures we used
EMBEDDING_DIM = 300
HIDDEN_SIZE = 1024
MAX_CHUNK_SIZE = 25


def preprocess():
    # Load images and captions
    X_train, X_valid, X_test, y_train, y_valid, y_test = data_loader(TOTAL_PIC, BATCH_SIZE)
    
    # Get all the distinct words and count the length of sentence
    chunk_size = []
    words_set = set()
    for caps in y_train:
        for i in range(caps.shape[0]):
            words_set = words_set.union(set(caps[i].split()))
            chunk_size.append(len(caps))
    chunk_size = np.sort(chunk_size)
    chunk_size = chunk_size[int(0.95*len(chunk_size))-1]
    chunk_size = min(chunk_size, MAX_CHUNK_SIZE)
    print('chunk size:', chunk_size)
    
    # Build a vocabulary dict
    vocab = {}
    vocab['<oov>']=len(vocab) 
    vocab['<start>'] = len(vocab)
    vocab['<pad>']=len(vocab)
    for word in words_set:
        vocab[word] = len(vocab)
    vocab_size = len(vocab)
    print('size of vocabulary:', vocab_size)
    # Build a reverse vocab dict
    rvocab = dict([(value, key) for (key, value) in vocab.items()])
    
    y_train_id_teacher = cap2id_teacher(y_train, BATCH_SIZE, vocab, chunk_size)
    y_valid_id_teacher = cap2id_teacher(y_valid, BATCH_SIZE, vocab, chunk_size)
    y_test_id_teacher = cap2id_teacher(y_test, BATCH_SIZE, vocab, chunk_size)
    
    y_train_id = cap2id(y_train, BATCH_SIZE, vocab, chunk_size)
    y_valid_id = cap2id(y_valid, BATCH_SIZE, vocab, chunk_size)
    y_test_id = cap2id(y_test, BATCH_SIZE, vocab, chunk_size)

    data = (X_train, X_valid, X_test, y_train, y_valid, y_test)
    y_teacher = (y_train_id_teacher, y_valid_id_teacher, y_test_id_teacher)
    y_id = (y_train_id, y_valid_id, y_test_id)
    book = (vocab, rvocab)
    return data, y_teacher, y_id, book


def train():
    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()
    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 1, "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    data, y_teacher, y_id, book = preprocess()
    X_train, X_valid, X_test, y_train, y_valid, y_test = data
    y_train_id_teacher, y_valid_id_teacher, y_test_id_teacher = y_teacher
    y_train_id, y_valid_id, y_test_id = y_id
    vocab, rvocab = book

    # Train lstm model
    model = caption_lstm(DROP_OUT, EMBEDDING_DIM, INPUT_CHANNEL, HIDDEN_SIZE, vocab_size, BATCH_SIZE*5)
    optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.95)
    train_loss = []
    valid_loss = []
    
    if os.path.exists('./model.pt'):
        checkpoint = torch.load('./model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)
    # parameters for training
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    cnn = Base_CNN()
    cnn = cnn.to(computing_device)
    for epoch in range(NUM_EPOCH):
        for i in range(len(X_train)):
            # len(X_train): 0.7 * TOTAL_PIC / BATCH_SIZE
            # X_train: (5*batch_size, 512, 196)
            scheduler.step()
            optimizer.zero_grad()
            # data: (5*batch_size, 512, 196)
            data = preprocessing(X_train[i].to(computing_device), cnn)
            cap = y_train_id[i].to(computing_device)
            cap_teacher = y_train_id_teacher[i].to(computing_device)
            pred, alpha_list = model(data, cap_teacher, chunk_size, computing_device)
            if (i%116==0) and (i!=0):
                print(torch.max(pred, dim=2)[1])
            # Regularize attention 
            alpha_all = torch.sum(alpha_list,dim=1)
            alpha_reg = torch.sum((chunk_size/196. - alpha_all) ** 2)
            # Get loss for backporp
            loss = criterion(pred.view(-1, vocab_size), cap.view(-1)) + alpha_reg
            train_loss.append(loss)
            loss.backward()
            optimizer.step()
            print('epoch:',epoch, 'batch:',i,'loss:',loss)
            
            with torch.no_grad():
                if i%7==0:
                    # data: (5*batch_size, 512, 196)
                    j = i % len(X_valid)
                    data = preprocessing(X_valid[j].to(computing_device), cnn)
                    cap = y_valid_id[j].to(computing_device)
                    cap_teacher = y_valid_id_teacher[j].to(computing_device)
                    pred, alpha_list = model(data, cap_teacher, chunk_size, computing_device)
                    # Regularize attention 
                    alpha_all = torch.sum(alpha_list,dim=1)
                    alpha_reg = torch.sum((chunk_size/196. - alpha_all) ** 2)
                    # Get loss for backporp
                    loss = criterion(pred.view(-1, vocab_size), cap.view(-1)) + alpha_reg
                    valid_loss.append(loss)
    
        # save model
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss':valid_loss
                    }, './model.pt')

    print("training done!")


if __name__ == '__main__':
    train()