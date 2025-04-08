from __future__ import annotations
from typing import List
import numpy as np

from autodiff.Tensor import Tensor

class MLP:
    def __init__(self, layers: List[Layer]):
        try:
            self.layers = []
            for layer in layers:
                self.add(layer)
        except Exception as e:
            raise e

    def add(self, layer: Layer):
        try:
            self.layers.append(layer)
        except Exception as e:
            raise e

    def get_params(self):
        try:
            all_params = []
            for layer in self.layers:
                if layer.has_params():
                    params = layer.get_params()
                    all_params.extend(params)
            return all_params
        except Exception as e:
            raise e

    def __call__(self, X: Tensor):
        try:
            self.forward(X)
        except Exception as e:
            raise e

    def forward(self, X: Tensor):
        try:
            layer_output = Tensor.copy(X)
            for layer in self.layers:
                print(layer_output.data.shape)
                layer_output = layer(layer_output)
                print(f"In layer {i}")
                i+=1
            return layer_output
        except Exception as e:
            raise e    

class Layer:
    pass

class Activation(Layer):
    def has_params():
        return False
    
class ParamLayer(Layer):
    def has_params():
        return True
    
class Linear(ParamLayer):
    def __init__(self, input_dim: int, layer_dim: int):
        self.input_dim = input_dim
        self.layer_dim = input_dim
        self.weights = Tensor(np.ones((layer_dim, input_dim)), store_grad=True)
        self.bias = Tensor(np.zeros(layer_dim), store_grad=True)

    def __call__(self, X: Tensor):
        try:
            self.forward(X)
        except Exception as e:
            raise e

    def get_params(self):
        try:
            return [self.weights, self.bias]
        except Exception as e:
            raise e
    
    def forward(self, X: Tensor):
        try:
            return (X @ self.weights) + self.bias
        except Exception as e:
            raise e

class ReLU(Activation):
    def __call__(self, x):
        return Tensor.reLU(x)
    
class Tanh(Activation):
    def __call__(self, x):
        return Tensor.tanh(x)

class Sigmoid(Activation):
    def __call__(self, x):
        return Tensor.sigmoid(x)
    
