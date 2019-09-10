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


def test():
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

    # Testing
    torch.cuda.empty_cache()
    model = caption_lstm(DROP_OUT, EMBEDDING_DIM, INPUT_CHANNEL, HIDDEN_SIZE, vocab_size, BATCH_SIZE*5)
    optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE)
    
    checkpoint = torch.load('./model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    valid_loss = checkpoint['valid_loss']
    
    model.to(computing_device)
    model.eval()
    cnn = Base_CNN()
    cnn = cnn.to(computing_device)
    
    test_loss = []
    with torch.no_grad():
        for i in range(len(X_test)):
            # len(X_train): 0.7 * TOTAL_PIC / BATCH_SIZE
            # X_train: (5*batch_size, 512, 196)
            # data: (5*batch_size, 512, 196)
            data = preprocessing(X_test[i].to(computing_device), cnn)
            cap = y_test_id[i].to(computing_device)
            cap_teacher = y_test_id_teacher[i].to(computing_device)
            pred, alpha_list = model(data, cap_teacher, chunk_size, computing_device)
            # Regularize attention 
            alpha_all = torch.sum(alpha_list,dim=1)
            alpha_reg = torch.sum((chunk_size/196. - alpha_all) ** 2)
            # Get loss for backporp
            loss = criterion(pred.view(-1, vocab_size), cap.view(-1)) + alpha_reg
            loss = criterion(pred.view(-1, vocab_size), cap.view(-1))
            test_loss.append(loss)
            print('epoch:',epoch, 'batch:',i,'loss:',loss)
    
    print('loss on test set:',np.mean([loss.item() for loss in test_loss]))
    
    # Generating
    torch.cuda.empty_cache()
    model = caption_lstm(DROP_OUT, EMBEDDING_DIM, INPUT_CHANNEL, HIDDEN_SIZE, vocab_size, BATCH_SIZE*5)
    optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE)
    
    checkpoint = torch.load('./model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    valid_loss = checkpoint['valid_loss']
    vocab = checkpoint['vocab']
    rvocab = checkpoint['rvocab']
    
    model.to(computing_device)
    model.eval()
    try_times=1
    
    cnn = Base_CNN()
    cnn = cnn.to(computing_device)
    with torch.no_grad():
        for k in range(try_times):
            print('Example', k+1)
            # Choose a random image
            i = random.randint(0, len(X_test)-1)
            j = random.randint(0, X_test[0].shape[0]-1)
        
            image = X_test[i][j, :, :, :].permute(1,2,0).numpy()
            print('Captions from human')
            print(y_test[i][j::BATCH_SIZE])
    
            data = preprocessing(X_test[i].to(computing_device), cnn)
            data = data[j,:,:].view(1, INPUT_CHANNEL, -1)
            pred, alpha_list = model.sampling(data, chunk_size, computing_device, rvocab, T=1)
            print('Captions from model')
            print(pred)
            visualize_att(image, alpha_list, pred)

    print("testing done!")


if __name__ == '__main__':
    test()