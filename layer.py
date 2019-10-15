import  torch
from    torch import nn
from    torch.nn import functional as F
from    utils import sparse_dropout, dot

import  gops


class GraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs

        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight

        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support


class DistributedGraphConvolution(GraphConvolution):
    def forward(self, inputs):
         # print('inputs:', inputs)
        x, supports = inputs

        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight

        comm_rank = gops.get_rank()
        comm_size = gops.get_size()

        feats = [xw, torch.zeros_like(xw)]

        current_feats = 0
        out = None

        for i in range(comm_size):
            gops.ring_pass(feats[current_feats].data_ptr(),
                    feats[current_feats ^ 1].data_ptr(),
                    feats[current_feats].shape.numel(), i)
            if out is None:
                out = torch.sparse.mm(supports[comm_rank], feats[current_feats])
            else:
                out += torch.sparse.mm(supports[comm_rank], feats[current_feats])
            current_feats ^= 1
            gops.wait()
        
        if self.bias is not None:
            out += self.bias

        return self.activation(out), supports
       
