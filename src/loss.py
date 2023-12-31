from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalLoss2(nn.Module):
    """
    this is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*(1-pt)*log(pt)
    Params:
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch. """
    def __init__(self, num_class, alpha=None, gamma=1, balance_index=-1, smooth=None, size_average=False):
        super(FocalLoss2, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
        if self.alpha is None:
           self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
    def forward(self, logit, target):

            #logit = F.softmax(input, dim=1)
            logit=torch.nn.functional.softmax(logit,dim=1)
            if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
                logit = logit.view(logit.size(0), logit.size(1), -1)
                logit = logit.permute(0, 2, 1).contiguous()
                logit = logit.view(-1, logit.size(-1))
            target = target.view(-1, 1)

            # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
            epsilon = 1e-6
            alpha = self.alpha.to(logit.device)

            idx = target.cpu().long()

            one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            one_hot_key = one_hot_key.to(logit.device)

            if self.smooth:
                one_hot_key = torch.clamp(one_hot_key, self.smooth / (self.num_class - 1), 1.0 - self.smooth)
            pt = (one_hot_key * logit).sum(1) + epsilon
            logpt = pt.log()
            gamma = self.gamma
            alpha = alpha[idx]
            loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss.sum()

            return loss