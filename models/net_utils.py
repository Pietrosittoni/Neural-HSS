import torch

def get_flops(model, inp, with_backward=False):
    from torch.utils.flop_counter import FlopCounterMode   
    istrain = model.training
    model.eval()
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)
    batch_size = float(inp.shape[0] )

    flop_counter = FlopCounterMode(display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops =  flop_counter.get_total_flops()
    # print(f'operatioons counted {flop_counter.get_flop_counts()}')
    if istrain:
        model.train()
    if hasattr(model, 'extra_flops'):
        total_flops += model.extra_flops
    return total_flops/batch_size,flop_counter



def count_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)