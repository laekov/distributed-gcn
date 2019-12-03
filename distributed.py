from layers import GraphConvolution
import sys


class Parallelizer(object):
    def __init__(self):
        self._support_shape = None

    def peep_graph(self, adj):
        self._support_shape = adj.shape

    def parallelize_layer(self, layer):
        raise NotImplementedError('Base parallelizer is abstract')


class NoParallel(Parallelizer):
    def __init__(self):
        super(NoParallel, self).__init__()

    def parallelize_layer(self, layer):
        return layer


class RowParallel(Parallelizer):
    def __init__(self, num_gpus):
        self._n = num_gpus
        super(RowParallel, self).__init__()

    def parallelize_layer(self, layer):
        raise NotImplementedError('Not implemented yet')
        return layer
