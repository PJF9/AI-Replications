import torch
from Value import Value


## Using Pytorch
x1 = torch.tensor([2.0]).double()    ; x1.requires_grad = True
x2 = torch.tensor([0.0]).double()    ; x2.requires_grad = True
w1 = torch.tensor([-3.0]).double()   ; w1.requires_grad = True
w2 = torch.tensor([1.0]).double()    ; w2.requires_grad = True
b = torch.tensor([6.88137]).double() ; b.requires_grad = True

n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

o.backward()

print("x1 grad", x1.grad.item())
print("w1 grad", w1.grad.item())
print("x2 grad", x2.grad.item())
print("w2 grad", w2.grad.item())


## Using Auto Grad
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
b = Value(6.88137, label="b")

n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

o.backward()

draw_dot(o)
