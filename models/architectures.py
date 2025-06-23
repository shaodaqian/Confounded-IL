from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

class MixtureGaussian(nn.Module):
    def __init__(self,hid_dim,input_channel,n_components):
        super(MixtureGaussian, self).__init__()
        self.input_channel=input_channel
        self.n_components=n_components
        self.mu=nn.Linear(hid_dim,input_channel*n_components)
        self.log_sig=nn.Linear(hid_dim,input_channel*n_components)
        self.pi=nn.Linear(hid_dim,input_channel*n_components)


    def forward(self,x):
        logits=F.softmax(self.pi(x).reshape(-1,self.input_channel,self.n_components),dim=-1)
        return (logits,self.mu(x).reshape(-1,self.input_channel,self.n_components),self.log_sig(x).reshape(-1,self.input_channel,self.n_components))



class FeedForward(nn.Module):
    def __init__(self, input_channel, hiddens,residual=False,dropout=0.1):
        super(FeedForward, self).__init__()
        # self.hiddens = hiddens

        self.FeedForward = nn.ModuleList()
        self.FeedForward.append(nn.Linear(input_channel, hiddens[0]))
        self.FeedForward.append(nn.BatchNorm1d(hiddens[0]))
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=dropout)
        self.FeedForward.append(nn.ReLU(True))
        self.FeedForward.append(nn.Dropout(p=dropout))
        for k in range(len(hiddens)-1):
            self.FeedForward.append(nn.Linear(hiddens[k], hiddens[k+1]))
            self.FeedForward.append(nn.BatchNorm1d(hiddens[k+1]))
            self.FeedForward.append(nn.ReLU(True))
            self.FeedForward.append(nn.Dropout(p=dropout))

        self.FeedForward = nn.Sequential(*self.FeedForward)

    def forward(self, input):
        return self.FeedForward(input)

class ConvNet(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, hiddens=[128,64,32],
            pool_size=(2, 2), kernel_size=(3, 3),dropout=0.1):
        super(ConvNet, self).__init__()
        # self.hiddens = hiddens

        self.Conv = nn.Sequential( #1x28x28
            nn.Conv2d(1,32, kernel_size=kernel_size,stride=1,padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size),
            nn.Dropout(p=dropout),
            nn.Conv2d(32, hiddens[1], kernel_size=kernel_size,stride=1,padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size),
            nn.Dropout(p=dropout),
            nn.Conv2d(hiddens[1], hiddens[1], kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Flatten(1,-1),
            # nn.Dropout(p=dropout),
            nn.Linear(hiddens[1]*9, hiddens[1]),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            # nn.Linear(hiddens[0], hiddens[1]),
            # nn.ReLU(True),
            # nn.Dropout(p=dropout),
        )

        self.FeedForward = nn.Sequential( #1x20x100
            nn.Linear(hiddens[1]+2, hiddens[2]),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        )

    def forward(self, input,other_features=[]):
        x_embedding=self.Conv(input.float())
        if len(other_features) > 0:
            embedd = torch.cat([x_embedding] + other_features, dim=1).float()
            # embedd = Concatenate(axis=1)([x_embedding] + other_features)
        else:
            embedd = x_embedding
        out=self.FeedForward(embedd)
        return out
