from Utils import draw_dot
from Value import Value

## Defining the expression L
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

e = a * b ; e.label = 'e'
d = e + c ; d.label = 'd'

f = Value(-2.0, label='f')
L = d * f ; L.label = 'L'

draw_dot(L)

## Backpropagation on a Neuron
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
b = Value(6.88137, label="b")

x1w1 = x1 * w1 ; x1w1.label = "x1*w1"
x2w2 = x2 * w2 ; x2w2.label = "x2*w2"
x1w1_x2w2 = x1w1 + x2w2 ; x1w1_x2w2.label = "x1*w1 + x2*w2"
n = x1w1_x2w2 + b ; n.label = "n"

# Using as activation function the `tanh`
o = n.tanh() ; o.label = 'o'

draw_dot(o)

o.backward()

draw_dot(o)
