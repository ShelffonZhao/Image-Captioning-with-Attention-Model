import pandas as pd
import pickle as pk
import torch
import numpy as np
import skimage.io as io
import os
from torchvision import transforms
from PIL import Image

def data_loader(sz = 100, bcsz = 5):
    p_train = 0.7
    p_val = 0.1

    imgs = None
    caps = None
    
    annotation_path = './dataset/results_20130124.token'
    annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])

    transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    if os.path.exists('data.pkl'):
        print("data.pkl found")
        with open('data.pkl', 'rb') as file:
            imgs = pk.load(file)
            caps = pk.load(file)
    else:
        print("data.pkl not found, trying to create")
        img_path= './dataset/flickr30k-images/*.jpg'
        imgs = io.ImageCollection(img_path)

        caps_path = './dataset/results_20130124.token'
        caps = pd.read_table(caps_path, sep='\t', header=None, names=['image', 'caption'])
        caps = caps.caption.values.reshape([-1, 5])
        

        with open('data.pkl', 'wb') as file:
            pk.dump(imgs, file)
            pk.dump(caps, file)

    tnsz = int(sz*p_train)
    vlsz = int(sz*p_val) + tnsz
    X_train = []
    y_train = []
    X_valid = []
    y_valid = []
    X_test = []
    y_test = []

    for i in range(0, sz, bcsz):
        img_bc = torch.stack([transform(Image.fromarray(imgs[j+i])) for j in range(bcsz)])
        
        cap_bc = []
        for j in range(i,i+bcsz):
            single_cap_idx = np.argwhere(annotations.image.values == imgs.files[j].split('/')[-1]+'#0')[0][0]//5
            #cap_idx.append(single_cap_idx)
            cap_bc.append(caps[single_cap_idx])

        
        #cap_bc = caps[cap_idx]
        if i<tnsz:
            X_train.append(img_bc)
            y_train.append(np.array(cap_bc))
        elif tnsz<=i<vlsz:
            X_valid.append(img_bc)
            y_valid.append(np.array(cap_bc))
        else:
            X_test.append(img_bc)
            y_test.append(np.array(cap_bc))
    for i in range(len(y_train)):
        y_train[i] = y_train[i].T.reshape(1,-1)[0]
    for i in range(len(y_test)):
        y_test[i] = y_test[i].T.reshape(1,-1)[0]
    for i in range(len(y_valid)):
        y_valid[i] = y_valid[i].T.reshape(1,-1)[0]
    return X_train, X_valid, X_test, y_train, y_valid, y_test
    