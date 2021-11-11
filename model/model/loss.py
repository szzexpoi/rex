import torch
import torch.nn.functional as F
from torch.autograd import Variable

epsilon = 1e-16

def cross_entropy(input,target,valid_mask):
	loss = -(target*torch.log(torch.clamp(input,min=epsilon,max=1))).sum(-1)
	loss = torch.mul(loss,valid_mask).sum(-1)/torch.clamp(valid_mask.sum(-1),min=1)
	return loss.mean()

def ans_cross_entropy(input,target):
	loss = -(target*torch.log(torch.clamp(input,min=epsilon,max=1))).sum(-1)
	return loss.mean()

def structure_bce(input,target):
	# balanced binary cross-entropy for structure gate prediction
	neg_weight = target.mean(-1,keepdim=True)
	pos_weight = 1-neg_weight
	loss = -(pos_weight*target*torch.log(torch.clamp(input,min=epsilon,max=1))+neg_weight*(1-target)*torch.log(torch.clamp(1-input,min=epsilon,max=1))).sum(-1)
	return loss.mean()
