import torch
import torch.nn as nn
import torch.nn.functional as F


class caption_lstm(nn.Module):
    def __init__(self, dropout, embedding_dim, channel, hidden_size, vocab_size, batch_size, num_layer=1):
        super(caption_lstm, self).__init__()
        self.embedding_dim = embedding_dim # 300
        self.channel = channel # 512
        self.input_size = self.channel + self.embedding_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self.project_fc = nn.Linear(self.channel, self.channel)
        self.init_h0_fc = nn.Linear(self.channel, self.hidden_size)
        self.init_c0_fc = nn.Linear(self.channel, self.hidden_size)
        self.Lo_fc = nn.Linear(self.embedding_dim, self.vocab_size)
        self.Lh_fc = nn.Linear(self.hidden_size*self.num_layer, self.embedding_dim)
        self.Lz_fc = nn.Linear(self.channel , self.embedding_dim)
        self.att_h_fc = nn.Linear(self.hidden_size, self.channel)
        self.att_w_fc = nn.Linear(self.channel, 1)
        self.beta_fc = nn.Linear(self.hidden_size*self.num_layer, 1)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layer, batch_first=True)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.input_bn = nn.BatchNorm1d(num_features=self.channel)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, a, y, num_steps, computing_device):
        # a: (batch_size, 512, 196)
        # y: (batch_size, num_steps)
        L = a.shape[2] # 196
        alpha_list = torch.zeros((self.batch_size, num_steps, L)).to(computing_device)
        pred = torch.zeros((self.batch_size, num_steps, self.vocab_size)).to(computing_device)
        h = self.init_h0(a)
        c = self.init_h0(a)
        y = self.embed_caption(y) # y: (batch_size, num_steps, embedding_dim)
        a = self.input_bn(a)
        a_proj = self.project_fc(a.permute(0,2,1)) # a_proj: (batch_size, 196, 512)
        for step in range(num_steps):
            z, alpha = self.get_attention(a, a_proj, h)
            # beta = self.get_beta(h)
            # z = beta * torch.sum(alpha * a, dim=2)
            # z: (batch_size, 512)
            x = torch.cat((y[:,step,:], z), dim=1).view(self.batch_size, 1, -1)
            output, (h,c) = self.lstm(x, (h,c))
            # Predict next y_t
            # pred: (batch_size, num_steps, vocab_size)
            pred[:, step, :] = self.get_pred(y, h, z)
            alpha_list[:,step,:] = alpha
        return pred, alpha_list
    
    def get_pred(self, y, h, z):
        # y: (batch_size, num_steps, embedding_dim)
        # h: (num_layer * num_directions, batch_size, hidden_size)
        # z: (batch_size, 512)
        h = self.dropout(h)
        h = h.view(self.batch_size, -1)
        h_logits = self.Lh_fc(h)
        # Add context vector and previous state later
        h_logits = self.tanh(h_logits)
        h_logits = self.dropout(h_logits)
        pred = self.Lo_fc(h_logits)
        # pred: (batch_size, vocab_size)
        return pred
        
    def get_attention(self, a, a_proj, h):
        # a: (batch_size, 512, 196)
        # a_proj: (batch_size, 196, 512)
        # h: (num_layer * num_directions, batch_size, hidden_size)
        L = a.shape[2] # 196
        h_att = self.relu(self.att_h_fc(h).view(self.batch_size,1,-1) + a_proj)
        out_att = self.att_w_fc(h_att.view(-1,self.channel)).view(-1,L)
        # alpha: (batch_size, 196)
        alpha = self.softmax(out_att) 
        # context: (batch_size, 512)
        context = torch.sum(a*alpha.view(self.batch_size, 1, L), dim=2)
        return context, alpha

    def get_beta(self, h):
        # h: (num_layer * num_directions, batch_size, hidden_size)
        return self.tanh(F.relu(self.beta_fc(h.view(self.batch_size, -1))))
        
    def embed_caption(self, cap_id):
        return self.embedding(cap_id)
    
    def init_h0(self, a):
        # a: (batch_size, 512, 196)
        # h0: (num_layer * num_directions, batch, hidden_size)
        h0 = self.tanh(self.init_h0_fc(torch.mean(a,dim=2)))
        h0 = h0.view(self.num_layer, self.batch_size, self.hidden_size)
        return h0

    def init_c0(self, a):
        # a: (batch_size, 512, 196)
        # c0: (num_layer * num_directions, batch, hidden_size)
        c0 = self.tanh(self.init_c0_fc(torch.mean(a,dim=2)))
        c0 = c0.view(self.num_layer, self.batch_size, self.hidden_size)
        return c0
    
    def sampling(self, a, num_steps, computing_device, rvocab, T):
        # a: (batch_size, 512, 196)
        # y: (batch_size, num_steps)
        L = a.shape[2]
        # Set batch_size = 1
        self.batch_size = 1
        # Initialize y
        y = torch.ones((1,1),dtype=torch.long).to(computing_device)
        # pred: (batch_size, num_steps, vocab_size)
        pred = torch.zeros((self.batch_size, num_steps, self.vocab_size)).to(computing_device)
        alpha_list = torch.zeros((self.batch_size, num_steps, L)).to(computing_device)
        h = self.init_h0(a)
        c = self.init_h0(a)
        a = self.input_bn(a)
        a_proj = self.project_fc(a.permute(0,2,1)) # a_proj: (batch_size, 196, 512)
        step = 0
        while step < num_steps:
            # pdb.set_trace()
            # y: (batch_size, num_steps, embedding_dim)
            y = self.embed_caption(y) 
            z, alpha = self.get_attention(a, a_proj, h)
            # beta = self.get_beta(h)
            # z = beta * torch.sum(alpha * a, dim=2) # z: (batch_size, 512)
            x = torch.cat((y[:,0,:], z), dim=1).view(self.batch_size, 1, -1)
            output, (h,c) = self.lstm(x, (h,c))
            # Predict next y_t
            pred[:,step,:] = self.get_pred(y, h, z) # (batch_size, vocab_size)
            y = torch.distributions.categorical.Categorical(self.softmax(pred[:,step,:]/T).view(-1)).sample().view(1, -1)
            alpha_list[:,step,:] = alpha
            step += 1
        pred = self.decode(pred, num_steps, rvocab)
        return pred, alpha_list
    
    def decode(self, pred, num_steps, rvocab):
        output = F.log_softmax(pred.view(num_steps, -1), dim=1)
        _, output = torch.max(output, dim=1)
        output = ' '.join([rvocab.get(i.item(), '?') for i in output])
        return output