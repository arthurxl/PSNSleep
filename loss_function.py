
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss,_WeightedLoss

from torch.nn.functional import normalize



class EucLoss(torch.nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self,):
        super(EucLoss, self).__init__()

    def loss_fn(self,x, y):
        x = normalize(x, dim=-1, p=2)
        y = normalize(y, dim=-1, p=2)
        distance=2 - 2 * (x * y).sum(dim=-1)
        return distance.sum()

    def euc_distance(self,x, y):
        x=normalize(x,dim=1)
        y = normalize(y, dim=1)
        distances = (x - y).pow(2).sum(1)
        return distances.sum()
    def forward(self, output1, output2):
        return self.loss_fn(output1,output2)




if __name__ == '__main__':
    x=torch.tensor(np.random.randint(0,10,[32,32,15])).to("cuda:0" if torch.cuda.is_available() else "cpu")
    y= torch.tensor(np.random.randint(0,10,[32,32,15])).to("cuda:0" if torch.cuda.is_available() else "cpu")
    print(x.shape)
    print(y.shape)
    loss_function=EucLoss()

    d=loss_function(x,y)
    print(len(d))

