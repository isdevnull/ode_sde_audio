from collections import OrderedDict


def compute_grad_norm(model):
    norm = 0
    for p in model.parameters():
        cur_grad = p.grad.square().sum() if p.grad is not None else 0
        norm += cur_grad
    
    return norm.cpu().sqrt()


def compute_weight_norm(model):
    norm = 0
    #norm_dict = OrderedDict()
    for n, p in model.named_parameters():
        cur_weight_norm = p.square().sum().cpu()
        #norm_dict[n] = cur_weight_norm.sqrt().cpu()
        norm += cur_weight_norm
    
    return norm.cpu().sqrt()