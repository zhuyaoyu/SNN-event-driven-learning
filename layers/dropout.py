import torch.nn as nn
import torch.nn.functional as f


class DropoutLayer(nn.Dropout3d):
    def __init__(self, config, name, inplace=False):
        self.name = name
        self.type = config['type']
        if 'p' in config:
            p = config['p']
        else:
            p = 0.5
        super(DropoutLayer, self).__init__(p, inplace)
        print(self.name)
        print("p: %.2f" % p)
        print("-----------------------------------------")

    def forward(self, x):
        ndim = len(x.shape)
        if ndim == 3:
            T, n_batch, N = x.shape
            result = f.dropout3d(x.permute(1,2,0).reshape((n_batch, N, 1, 1, T)), self.p, self.training, self.inplace)
            return result.reshape((n_batch, N, T)).permute(2,0,1)
        elif ndim == 5:
            T, n_batch, C, H, W = x.shape
            result = f.dropout3d(x.permute(1,2,3,4,0).reshape((n_batch, C * H, W, 1, T)), self.p, self.training, self.inplace)
            return result.reshape((n_batch, C, H, W, T)).permute(4,0,1,2,3)
        else:
            raise("In dropout layer, dimension of input is not 3 or 5!")

    def weight_clipper(self):
        return