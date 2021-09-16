import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

#This code is based on https://github.com/zfjsail/gae-pytorch and https://github.com/GRAND-Lab/ARGA

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar
    
class GCNModelAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj)

    def forward(self, x, adj):
        z = self.encode(x, adj)
        return self.dc(z), z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class Regularizer(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Regularizer, self).__init__()
        self.dc_den1 = torch.nn.Linear(hidden_dim2, hidden_dim3)
        self.dc_den1.bias.data.fill_(0.0)
        self.dc_den1.weight.data = torch.normal(0.0, 0.001, [hidden_dim3, hidden_dim2])
        self.dc_den2 = torch.nn.Linear(hidden_dim3, hidden_dim1)
        self.dc_den2.bias.data.fill_(0.0)
        self.dc_den2.weight.data = torch.normal(0.0, 0.001, [hidden_dim1, hidden_dim3])
        self.dc_output = torch.nn.Linear(hidden_dim1, 1)
        self.dc_output.bias.data.fill_(0.0)
        self.dc_output.weight.data = torch.normal(0.0, 0.001, [1,hidden_dim1])
        self.act = torch.sigmoid
    def forward(self, inputs):
        dc_den1 = self.act(self.dc_den1(inputs))
        dc_den2 = torch.sigmoid((self.dc_den2(dc_den1)))
        output = self.dc_output(dc_den2)
        return output