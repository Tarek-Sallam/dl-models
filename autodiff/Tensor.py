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
    
    def __init__(self, data: ArrayLike, store_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data) if store_grad else None
        self.store_grad = store_grad
        self._prev = set()
        self._backward = lambda : None
        self._op = ''

    def __add__(self, other: Union[Tensor, float]) -> Tensor:
        try:
            other = self.convertToTensor(other)
            out = Tensor(self.data + other.data, store_grad=self.store_grad or other.store_grad)
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

    def backward(self):
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