import torch
from torch import nn

from utils import problem


class ReLULayer(nn.Module):
    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a Rectified Linear Unit calculation (ReLU):
        Element-wise:
            - if x > 0: return x
            - else: return 0

        Args:
            x (torch.Tensor): More specifically a torch.FloatTensor, with some shape.
                Input data.

        Returns:
            torch.Tensor: More specifically a torch.FloatTensor, with the same shape as x.
                Every negative element should be substituted with 0.
                Output data.

        Note:
            - YOU ARE NOT ALLOWED to use torch.nn.ReLU (or it's functional counterparts) in this class
            - Make use of pytorch documentation: https://pytorch.org/docs/stable/index.html
        """
        # torch.zeros_like and torch.maximum
        zeros = torch.zeros_like(x) #using zeros_like function to create a vector of zeros that is of the same size as x, source: https://docs.pytorch.org/docs/stable/generated/torch.zeros_like.html
        relu = torch.maximum(x,zeros) #using maximum to function to evaluate element-wise whether x_i or 0 is the maximum for relu, source: https://docs.pytorch.org/docs/stable/generated/torch.maximum.html
        return relu
        raise NotImplementedError("Your Code Goes Here")
