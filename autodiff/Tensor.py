from __future__ import annotations
import numpy as np
from typing import Union, List

ArrayLike = Union[float, int, List[float], List[int], np.ndarray]

class Tensor:

    @staticmethod
    def convertToTensor(data):
        if not isinstance(data, Tensor):
            try:
                data = Tensor(data)
                return data
            except Exception as e:
                print("Could not convert to Tensor")
                raise e
        else:
            return data
    
    def __init__(self, data: ArrayLike, store_grad:bool =False):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data) if store_grad else None
        self.store_grad = store_grad
        self._prev = set()
        self._backward = lambda : None
        self._op = ''

    def copy(self, shallow=False) -> Tensor:
        try:
            if shallow:
                data = self.data
            else:
                data = self.data.copy()

            out = Tensor(data, store_grad=self.store_grad)
            return out
        except Exception as e:
            raise e
        
    def __add__(self, other: Union[ArrayLike, Tensor]) -> Tensor:
        try:
            other = self.convertToTensor(other)
            result = self.data + other.data
            out = Tensor(result, store_grad=self.store_grad or other.store_grad)
            out._prev = {self, other}
            out._op = 'add'

            def _backward():
                if self.store_grad:
                    self.grad += out.grad
                if other.store_grad:
                    other.grad += out.grad

            out._backward = _backward

            return out
        except Exception as e:
            raise e
    
    __radd__ = __add__
        
    def __mul__(self, other: Union[ArrayLike, Tensor]) -> Tensor:
        try:
            other = self.convertToTensor(other)
            result = self.data * other.data
            out = Tensor(result, store_grad=self.store_grad or other.store_grad)
            out._prev = {self, other}
            out._op = 'mul'

            def _backward():
                if self.store_grad:
                    self.grad += out.grad * other.data
                if other.store_grad:
                    other.grad += out.grad * self.data
            
            out._backward = _backward
            return out
        except Exception as e:
            raise e
    __rmul__ = __mul__
        
    def __truediv__(self, other: Union[ArrayLike, Tensor]) -> Tensor:
        try:
            other = self.convertToTensor(other)
            result = 1 / other.data
            out1 = Tensor(result, store_grad=other.store_grad)
            out1._prev = {other}
            out1._op = 'div'
            def _backward():
                if other.store_grad:
                    other.grad += - out1.grad*(other.data**2)
            out1._backward = _backward
            return self.__mul__(self, out1)
    
        except Exception as e:
            raise e
    
    def __rtruediv__(self, other: Union[ArrayLike, Tensor]) -> Tensor:
        try:
            self.convertToTensor(other)
            return other.__truediv__(self)
            
        except Exception as e:
            raise e

    def __matmul__(self, other: Union[ArrayLike, Tensor]) -> Tensor:
        try:
            other = self.convertToTensor(other)
            result = self.data @ other.data
            out = Tensor(result, store_grad=self.store_grad or other.store_grad)
            out._prev = {self, other}
            out._op = 'matmul'

            def _backward():
                if self.store_grad:
                    self.grad += out.grad * other.data.T
                if other.store_grad:
                    other.grad += out.grad * self.data.T
            
            out._backward = _backward
            return out
        
        except Exception as e:
            raise e
        
    def __rmatmul__(self, other: Union[ArrayLike, Tensor]) -> Tensor:
        try:
            self.convertToTensor(other)
            return other.__matmul__(self)
            
        except Exception as e:
            raise e

    @staticmethod
    def reLU(tensor: Union[ArrayLike, Tensor]) -> Tensor:
        try:
            tensor = Tensor.convertToTensor(tensor)
            result = np.clip(tensor.data, a_min=0)
            out = Tensor(result, store_grad=tensor.store_grad)
            out._prev={tensor}
            out._op = 'reLU'
            
            def _backward():
                tensor.grad += out.grad * (out.data > 0).astype(np.float32)

            out._backward = _backward
            return out
        
        except Exception as e:
            raise e
        
    @staticmethod
    def tanh(tensor: Union[ArrayLike, Tensor]) -> Tensor:
        try:
            tensor = Tensor.convertToTensor(tensor)
            result = np.tanh(tensor.data)
            out = Tensor(result, store_grad=tensor.store_grad)
            out._prev={tensor}
            out._op = 'tanh'

            def _backward():
                tensor.grad += out.grad * (1 - out.data**2)

            out._backward = _backward
            return out
        except Exception as e:
            raise e
        
    @staticmethod
    def sigmoid(tensor: Union[ArrayLike, Tensor]) -> Tensor:
        try:
            tensor = Tensor.convertToTensor(tensor)
            result = 1 / (1 + np.e**-tensor.data)
            out = Tensor(result, store_grad=tensor.store_grad)
            out._prev={tensor}
            out._op = 'sigmoid'

            def _backward():
                tensor.grad += out.grad * (out.data)*(1 - out.data)

            out._backward = _backward
            return out
        except Exception as e:
            raise e
        
    @staticmethod
    def log2(tensor: Union[ArrayLike, Tensor]) -> Tensor:
        try:
            tensor = Tensor.convertToTensor(tensor)
            result = np.log2(tensor.data)
            out = Tensor(result, store_grad=tensor.store_grad)
            out._prev={tensor}
            out._op = 'log2'

            def _backward():
                tensor.grad += out.grad * 1/tensor.data * np.log2(np.e)

            out._backward = _backward
            return out
        except Exception as e:
            raise e
        
    @staticmethod
    def log(tensor: Union[ArrayLike, Tensor]) -> Tensor:
        try:
            tensor = Tensor.convertToTensor(tensor)
            result = np.log(tensor.data)
            out = Tensor(result, store_grad=tensor.store_grad)
            out._prev={tensor}
            out._op = 'log'

            def _backward():
                tensor.grad += out.grad * 1/tensor.data

            out._backward = _backward
            return out
        except Exception as e:
            raise e
        
    @staticmethod
    def sum(tensor: Union[ArrayLike, Tensor], axis=0) -> Tensor:
        try:
            tensor = Tensor.convertToTensor(tensor)
            result = np.sum(tensor.data, axis=axis)
            out = Tensor(result, store_grad=tensor.store_grad)
            out._prev={tensor}
            out._op = 'sum'

            def _backward():
                tensor.grad += out.grad

            out._backward = _backward
            return out
        except Exception as e:
            raise e
        
    @staticmethod
    def mean(tensor: Union[ArrayLike, Tensor], axis=0) -> Tensor:
        try:
            tensor = Tensor.convertToTensor(tensor)
            n = tensor.data.shape[axis]
            return Tensor.sum(tensor) / Tensor(n)
        
        except Exception as e:
            raise e
        
    def backward(self, weight: ArrayLike = None):
        try:
            if weight is not None:
                if weight.shape != self.data.shape:
                    raise ValueError("Shape of weight does not match shape of tensor")
                self.grad = np.array(weight)
            else:
                self.grad = np.ones_like(self.data)
                    
            topo = []
            visited = set()

            def build_topo(tensor):
                if tensor not in visited:
                    visited.add(tensor)
                    for prev in tensor._prev:
                        build_topo(prev)
                    topo.append(tensor)
            
            build_topo(self)
            for tensor in reversed(topo):
                tensor._backward()
        except Exception as e:
            raise e