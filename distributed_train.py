import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F

import  numpy as np
from    data import load_data, preprocess_features, preprocess_adj
from    model import DistributedGCN
from    config import  args
from    utils import masked_loss, masked_acc

import time

import gops

comm_rank = gops.get_rank()
comm_size = gops.get_size()

print(comm_rank, comm_size)

def main():
    gops.init()

    seed = 123
    np.random.seed(seed)
    torch.random.manual_seed(seed)


    # load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)

    # D^-1@X
    features = preprocess_features(features) # [49216, 2], [49216], [2708, 1433]

    device = torch.device('cuda')
    train_label = torch.from_numpy(y_train).long().to(device)
    num_classes = train_label.shape[1]
    train_label = train_label.argmax(dim=1)
    train_mask = torch.from_numpy(train_mask.astype(np.int)).to(device)
    val_label = torch.from_numpy(y_val).long().to(device)
    val_label = val_label.argmax(dim=1)
    val_mask = torch.from_numpy(val_mask.astype(np.int)).to(device)
    test_label = torch.from_numpy(y_test).long().to(device)
    test_label = test_label.argmax(dim=1)
    test_mask = torch.from_numpy(test_mask.astype(np.int)).to(device)

    i = torch.from_numpy(features[0]).long().to(device)
    v = torch.from_numpy(features[1]).to(device)
    feature = torch.sparse.FloatTensor(i.t(), v, features[2]).to(device)

    support = []
    for a in adj:
        acoo = a.tocoo()
        i = torch.tensor([acoo.row, acoo.col], dtype=torch.long)
        v = torch.tensor(acoo.data, dtype=torch.float32)
        support.append(torch.sparse.FloatTensor(i, v, acoo.shape).float().to(device))

    print('x :', feature)
    print('sp:', support)
    num_features_nonzero = feature._nnz()
    feat_dim = feature.shape[1]


    net = DistributedGCN(feat_dim, num_classes, num_features_nonzero)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    times = {'fw': [], 'bw': []}

    net.train()
    for epoch in range(args.epochs):
        t_beg = time.time()
        out = net((feature, support))
        out = out[0]
        loss = masked_loss(out, train_label, train_mask)
        loss += args.weight_decay * net.l2_loss()
        t_loss = time.time()

        acc = masked_acc(out, train_label, train_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t_end = time.time()

        times['fw'].append(t_loss - t_beg)
        times['bw'].append(t_end - t_loss)

        if epoch % 10 == 0:

            print('Epoch {} Loss {:.5f} Acc {:.4f} T_fw {:.6f} T_bw {:.6f}'.format(
                epoch, loss.item(), acc.item(), 
                np.mean(times['fw'][-10:-1]), 
                np.mean(times['bw'][-10:-1])))

    net.eval()

    out = net((feature, support))
    out = out[0]
    acc = masked_acc(out, test_label, test_mask)
    print('test:', acc.item())


if __name__ == '__main__':
    main()

