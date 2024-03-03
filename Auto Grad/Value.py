import math


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        """
        `_children` (tuple): For a `Value` object tell us the `Value` objects that produce it.
        `_op` (str): The operation that from children we can produce `self`.
        `label` (str): The label of the object, for visualizing the expression.
        """

        self.data = data
        self.grad = 0.0               # Initializing to 0 because we don't want to affect the output
        self._backward = lambda: None # Initializing to empy function to handle the leaf nodes
        self._prev = set(_children)   # We are storing in a set for efficiency, O(logn) retrieving and adding complexity
        self._op = _op
        self.label = label


    def __repr__(self):
        if self.label != '':
            return f"Value=({self.data}, label={self.label})"
        return f"Value=({self.data})"
    

    def __add__(self, other):
        other = Value(other) if not isinstance(other, Value) else other # handling the case where we want Value + int

        out = Value(self.data + other.data, _children=(self, other), _op='+')

        # Updating `_children` total gradients by passing through derivative
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    

    def __radd__(self, other):
        return self + other # handling the case: int + Value
    

    def __sub__(self, other): # Value - other
        return self + (-other)


    def __mul__(self, other):
        other = Value(other) if not isinstance(other, Value) else other # handling the case where we want Value * int
        out = Value(self.data * other.data, _children=(self, other), _op='*')

        # Updating `_children` total gradients by applying the rule we found out before
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    

    def __rmul__(self, other):
        return self * other # handling the case: int * Value
    

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float types"

        out = Value(self.data**other, _children=(self,), _op=f"**{other}")

        def backward():
            self.grad += other*(self.data**(other - 1)) * out.grad

        out._backward = backward

        return out
    

    def __truediv__(self, other): # for Value / Value
        return self * (other ** -1)
    

    def exp(self):
        out = Value(math.exp(self.data), _children=(self, ), _op="exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out


    def tanh(self):
        out = Value((math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1), _children=(self,), _op="tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out
    
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for n in reversed(topo):
            n._backward()
