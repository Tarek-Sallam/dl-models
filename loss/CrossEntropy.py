import numpy as np
from autodiff.Tensor import Tensor
from typing import Literal

class BCELoss():
    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean"):
        self.reduction = reduction
        self.output = None

    def __call__(self, y: Tensor, y_pred: Tensor):
        self.forward(y, y_pred)

    def forward(self, y: Tensor, y_pred: Tensor):
        loss = -(y * Tensor.log(y_pred) + (1 - y)*(Tensor.log((1 - y_pred))))
        if self.reduction == 'none':
            self.output = loss
        elif self.reduction == 'sum':
            self.output = Tensor.sum(loss)
        elif self.reduction == 'mean':
            self.output = Tensor.mean(loss)
        else:
            raise ValueError(f'Invalid reduction value: {self.reduction}')
        
        return self.output
        
    def backward(self):
        self.output.backward()