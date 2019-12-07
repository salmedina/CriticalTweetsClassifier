from .functional import revgrad
from torch.nn import Module


class RevGrad(Module):
    def __init__(self, scale=1.0, *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """

        super().__init__(*args, **kwargs)
        self.scale = scale


    def forward(self, input_):
        RevGrad.scale = self.scale
        return revgrad(input_)