import matplotlib.pyplot as plt
from skimage.transform import resize

import torch
import torch.nn as nn
from torchvision import models

# Data preprocessing
# Model for pre-trained VGG
class Base_CNN(nn.Module):
    def __init__(self):
        super(Base_CNN,self).__init__()
        vgg = models.vgg16(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = False
        vgg_module = list(vgg.children())[0][0:30]
        # print(vgg_module)
        self.vgg_net = nn.Sequential(*vgg_module)
        
    def forward(self, images):
        output = self.vgg_net(images)
        return output
      

def preprocessing(images, cnn):
    #input: batch_size, 3, 224,224
    #ouput: batch_size*5, 512,196
    out = cnn(images)
    out_size = out.size()
    out = out.view(out_size[0],out_size[1],-1)
    out = torch.cat((out,out,out,out,out),0)
    return out


# Get ready for embedding of captions
def cap2id_teacher(y_xxx, batch_size, vocab, chunk_size):
    # len(y_xxx): len(X_xxx)
    # y_xxx[i].shape: (5*batch_size,)
    text = []
    for sents in y_xxx:
        sub_text = torch.zeros((5*batch_size, chunk_size), dtype=torch.long)
        for i in range(sents.shape[0]):
            cap = []
            for word in ['<start>']+sents[i].split():
                if word in vocab:
                    cap.append(vocab[word])
                else:
                    cap.append(vocab['<oov>'])
            cap = cap[:chunk_size]
            if len(cap) < chunk_size:
                cap += [2]*(chunk_size-len(cap))
            sub_text[i, :] = torch.tensor(cap, dtype=torch.long)
        text.append(sub_text)
    # len(text): len(X_xxx)
    # len(text[i]): 25
    # text[i][j]: index of words in caption for one image
    return text


def cap2id(y_xxx, batch_size, vocab, chunk_size):
    # len(y_xxx): len(X_xxx)
    # y_xxx[i].shape: (5*batch_size,)
    text = []
    for sents in y_xxx:
        sub_text = torch.zeros((5*batch_size, chunk_size), dtype=torch.long)
        for i in range(sents.shape[0]):
            cap = []
            for word in sents[i].split():
                if word in vocab:
                    cap.append(vocab[word])
                else:
                    cap.append(vocab['<oov>'])
            cap = cap[:chunk_size]
            if len(cap) < chunk_size:
                cap += [2]*(chunk_size-len(cap))
            sub_text[i, :] = torch.tensor(cap, dtype=torch.long)
        text.append(sub_text)
    # len(text): len(X_xxx)
    # len(text[i]): 25
    # text[i][j]: index of words in caption for one image
    return text


def visualize_att(image, alpha_list, pred):
    alpha_list = alpha_list.detach().cpu().numpy()
    pred = pred.split()
    image = (image-image.min())/(image.max()-image.min())*255
    plt.figure()
    plt.imshow((image).astype(int))
    plt.title('original title')
    plt.show()
    for step in range(alpha_list.shape[1]):
        if pred[step] == '<pad>':
            continue
        alpha = alpha_list[:,step,:].reshape((14,14,1))
        alpha = (1-(alpha - alpha.min())/(alpha.max()-alpha.min()))/1
        alpha = resize(alpha,(224, 224),anti_aliasing=True)
        plt.figure()
        plt.imshow((image*alpha).astype(int))
        plt.title(pred[step])
        plt.show()