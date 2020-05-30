from torch.autograd import Function


class RevGrad(Function):
    scale = 1.0
    @staticmethod
    def forward(ctx, input_):
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.neg() #* RevGrad.scale
        return grad_input


revgrad = RevGrad.apply