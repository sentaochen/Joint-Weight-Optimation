import torch
import torch.nn.functional as F
import torch.nn as nn

def entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def weighted_cross_entropy(out,label,weight=None):
    if weight is not None:
        cross_entropy = F.cross_entropy(out,label,reduction='none')
        return torch.sum(weight*cross_entropy)/(torch.sum(weight)+1e-5)
    else:
        return F.cross_entropy(out,label)
