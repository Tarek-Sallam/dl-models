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