import scipy.sparse as sp
import networkx as nx
import argparse
from data import load_data, preprocess_features, preprocess_adj
import pickle


parser = argparse.ArgumentParser(description='Preprocess GCN dataset')
parser.add_argument('--dataset', default='cora')
parser.add_argument('-n', type=int, default=4)
args = parser.parse_args()


def main():
    objs, test_idxs, adj = load_data(args.dataset, mode='preprocess')
    test_idxs = list(test_idxs)

    tot_sz = adj.shape[0]
    adj = preprocess_adj(adj)
    rid = [c[0] for c in adj[0]]
    cid = [c[1] for c in adj[0]]
    adj = sp.csr_matrix((adj[1], (rid, cid)), shape=(tot_sz, tot_sz))
    n = args.n
    chunk_sz = tot_sz // n
    print('{} * {} = {}'.format(n, chunk_sz, tot_sz))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally']
    for chunk_id in range(n):
        lb = chunk_id * chunk_sz
        ub = lb + chunk_sz
        prefix = 'data/ind.{}_{}_of_{}'.format(args.dataset, chunk_id, n)
        for name, obj in zip(names, objs):
            with open('{}.{}'.format(prefix, name), 'wb') as f:
                pickle.dump(obj[lb:ub], f)
        sadj = [adj[lb:ub, i * chunk_sz:(i + 1) * chunk_sz] for i in range(n)]
        with open('{}.graph'.format(prefix), 'wb') as f:
            pickle.dump(sadj, f)
        with open('{}.test.index'.format(prefix), 'w') as f:
            written = False
            while len(test_idxs) > 0 and test_idxs[0] < ub:
                f.write('{}\n'.format(test_idxs.pop(0) - lb))
                written = True
            if not written:
                f.write('{}\n'.format(chunk_sz - 1))
        print('{} of {} finished'.format(chunk_id, n))

if __name__ == '__main__':
    main()
