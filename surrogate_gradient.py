import torch


class ExponentialSurroGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.ge(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input * torch.exp(-torch.abs(input))


class TriangleSurroGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma=1.0):
        out = input.ge(0.)
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, L)
        return out.float()

    @staticmethod
    def backward(ctx, grad_output):
        (input, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


