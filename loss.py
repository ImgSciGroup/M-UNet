import torch
import torch.nn.functional as F
from args import args
def focal_loss(y_true, y_pred):
    alpha, gamma = 0.25, 2
    p = torch.sigmoid(y_pred)
    return torch.sum(- alpha*(y_true * torch.log(p) * ((1 - p)**gamma)+(1 - y_true) * torch.log(1 - p) *(p**gamma)))/((args.patch_size** 2)*args.batch_size)
# def wbce_loss(y_pred, label ,alpha):
#    # 计算loss
#    p = torch.sigmoid(y_pred)
#
#    # p = torch.clamp(p, min=1e-9, max=0.99)
#
#    # loss = torch.sum(torch.abs_(-alpha*label*torch.log(p)+(1-label)*torch.log(1-p))) / len(label)
#    loss = torch.sum(- alpha * torch.log(p) * label\
#                     - torch.log(1 - p) * (1 - label)) / ((128 ** 2)*8)
#    return loss
def dice_loss(pred,label):
    pred=torch.sigmoid(pred)
    loss=torch.sum(1-2*pred*label/(pred+label)) / ((args.patch_size ** 2)*args.batch_size)
    return loss