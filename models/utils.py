import torch

def get_flops(model, inp, with_backward=False):
    from torch.utils.flop_counter import FlopCounterMode    
    istrain = model.training
    model.eval()
    
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops



def count_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)