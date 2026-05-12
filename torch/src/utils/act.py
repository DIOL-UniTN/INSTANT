import torch

class TSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T):
        x = x / T  # Scale logits
        exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])  # Stability trick
        softmax_out = exp_x / exp_x.sum(dim=1, keepdim=True)
        ctx.save_for_backward(softmax_out, T)  # Save for backward pass
        return softmax_out

    @staticmethod
    def backward(ctx, grad_output):
        softmax_out, T = ctx.saved_tensors
        grad_input = softmax_out * (grad_output - (grad_output * softmax_out).sum(dim=-1, keepdim=True))
        grad_T = None
        return grad_input / T, grad_T # Gradient w.r.t. input
