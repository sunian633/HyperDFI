import torch as t
import torch.nn.functional as F


def compute_reg_loss(model):
    ret = 0
    for weight in model.parameters():
        ret += weight.norm(2).square()
    return ret


def cross_entropy_loss(pred, target):
    return F.cross_entropy(pred, target)


def compute_contrast_loss(embeds1, embeds2, nodes, temp):
    embeds1 = F.normalize(embeds1 + 1e-8, p=2)
    embeds2 = F.normalize(embeds2 + 1e-8, p=2)
    picked_embeds1 = embeds1[nodes]
    picked_embeds2 = embeds2[nodes]
    numerator = t.exp(t.sum(picked_embeds1 * picked_embeds2, dim=-1) / temp)
    denominator = t.exp(picked_embeds1 @ embeds2.T / temp).sum(-1) + 1e-8
    return -t.log(numerator / denominator).mean()


def row_l2_normalize(x):
    epsilon = t.FloatTensor([1e-12]).cuda()
    return x / (t.max(t.norm(x, dim=1, keepdim=True), epsilon))
