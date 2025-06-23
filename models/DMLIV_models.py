import torch
import torch.nn as nn
from architectures import FeedForward, ConvNet,MixtureGaussian

from utils import random_gmm

class MLP(nn.Module):
    '''
    Adds sampling functionality to a Keras model and extends the losses to support
    mixture of gaussian losses.

    # Argument
    '''

    def __init__(self, input_channel, hiddens, output_channel=1, dropout=0.05, use_image=False, image_shape=None):
        super(MLP, self).__init__()
        self.use_image = use_image
        self.image_shape = image_shape
        if use_image:
            self.net = ConvNet(hiddens, dropout=dropout)
        else:
            self.net = FeedForward(input_channel, hiddens, dropout)
        self.output = nn.Linear(hiddens[-1], output_channel)

    def forward(self, x):
        x=x.clone()
        if self.use_image:
            fnn = self.net(x)
        else:
            fnn = self.net(x)
        out = self.output(fnn)
        return out

class MixtureGaussian_MLP(nn.Module):
    '''
    Adds sampling functionality to a Keras model and extends the losses to support
    mixture of gaussian losses.

    # Argument
    '''

    def __init__(self, input_channel,hiddens,output_channel=1,dropout=0.01,n_components=5,use_image=False,image_shape=None):
        super(MixtureGaussian_MLP, self).__init__()
        self.image_shape=image_shape
        self.use_image=use_image
        self.n_components=n_components
        if use_image:
            self.net=ConvNet(hiddens,dropout=dropout)
        else:
            self.fnn=FeedForward(input_channel,hiddens,dropout)
        self.gaussian=MixtureGaussian(hiddens[-1],output_channel,n_components)

    def forward(self,x):
        if self.use_image:
            # time, image = feat[:, 0:1].float(), feat[:, 1:].float()
            # image=image.reshape(self.image_shape)
            # fnn = self.net(image, [time, inst])
            fnn=self.net(x)
        else:
            fnn=self.fnn(x)
        out=self.gaussian(fnn)
        # out shaped batch,dim,ncomponent
        return out

    def sample(self,x):
        if self.use_image:
            fnn = self.net(x)
        else:
            fnn=self.fnn(x)
        out=self.gaussian(fnn)
        [pi, mu, log_sig] = out
        samples = random_gmm(pi, mu, torch.exp(log_sig),self.n_components)
        return samples


