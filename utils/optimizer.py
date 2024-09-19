# -*- coding: utf-8 -*-
import torch.optim as optim
def get_optimizer(model, init_lr, lr_mult, momentum = 0.9, weight_decay = 0.0005, step=None):
    gamma = 0.0001
    power = 0.75
    init_lr = init_lr * (1 + gamma * step) ** (-power)
    w_finetune, b_finetune, w_train, b_train = [], [], [], []
    for k, v in model.named_parameters():
        if k.__contains__('fc') or k.__contains__('bottleneck'):
            if k.__contains__('weight'):
                w_train.append(v)
            else:
                b_train.append(v) 
        else:
            if k.__contains__('weight'): # or k.__contains__('domain_classifier')
                w_finetune.append(v)
            else:
                b_finetune.append(v) 
    optimizer = optim.SGD([{'params': w_finetune, 'lr': init_lr * lr_mult[0]},
                               {'params': b_finetune, 'lr': init_lr * lr_mult[1]},
                               {'params': w_train, 'lr': init_lr * lr_mult[2]},
                               {'params': b_train, 'lr': init_lr * lr_mult[3]}],
                              lr = init_lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    return optimizer  
