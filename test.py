from autodiff.Tensor import Tensor

a = Tensor(4.0, store_grad=True)
b = Tensor(2.0, store_grad=True)
c = a*b
d = 2*c*a
d.backward()
print(a.grad)
print(b.grad)